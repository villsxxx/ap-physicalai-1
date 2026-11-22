"""Sequential training/evaluation campaign. Run manually in the existing GPU container."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parent


def now():
    return datetime.now(timezone.utc).isoformat()


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)


def load_json(path, fallback):
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else fallback


def parse_training(text):
    rewards = re.findall(r'Mean reward:\s*([-+\d.eE]+)', text)
    seconds = re.findall(r'Обучение завершено за\s+(\d+)\s+с', text)
    return {'mean_reward': float(rewards[-1]) if rewards else None,
            'reported_seconds': int(seconds[-1]) if seconds else None}


def parse_inference(text):
    match = re.search(r'Итого: захватов (\d+)/(\d+), success rate = (\d+)/(\d+), средняя награда ([-+\d.eE]+)', text)
    if not match:
        raise ValueError('Итоговая строка инференса отсутствует')
    grasps, grasp_n, successes, episodes = map(int, match.groups()[:4])
    if episodes != 10 or grasp_n != episodes or not 0 <= successes <= episodes or not 0 <= grasps <= episodes:
        raise ValueError('Неполный или некорректный инференс')
    return {'grasps': grasps, 'successes': successes, 'episodes': episodes,
            'mean_inference_reward': float(match[5])}


def retained_updates(segment, checkpoint_iteration):
    count = checkpoint_iteration - segment['start_iteration'] + 1
    if count <= 0 or count > segment['requested_updates']:
        raise ValueError('Чекпоинт не соответствует диапазону текущего сегмента')
    return segment['retained_before'] + count


def stream(command, log_path):
    print('\n$ ' + ' '.join(map(str, command)), flush=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    env = dict(os.environ, PYTHONUNBUFFERED='1')
    process = subprocess.Popen(list(map(str, command)), cwd=ROOT, env=env,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               stdin=subprocess.DEVNULL, text=True,
                               encoding='utf-8', errors='replace')
    try:
        with log_path.open('w', encoding='utf-8') as log:
            for line in process.stdout:
                print(line, end='', flush=True)
                log.write(line)
                log.flush()
        return process.wait(), round(time.monotonic() - started, 2)
    except BaseException:
        process.terminate()
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        raise


def finite_checkpoint(path):
    import torch
    try:
        data = torch.load(path, map_location='cpu', weights_only=False)
        tensors = 0

        def check(value):
            nonlocal tensors
            if isinstance(value, torch.Tensor):
                tensors += 1
                return bool(torch.isfinite(value).all())
            if isinstance(value, dict):
                return all(check(v) for v in value.values())
            if isinstance(value, (list, tuple)):
                return all(check(v) for v in value)
            return True

        if not check(data) or tensors == 0 or 'iter' not in data:
            return None
        return int(data['iter'])
    except Exception as error:
        print(f'Чекпоинт пропущен: {path.name}: {error}', flush=True)
        return None


def latest_finite(segment):
    directory = ROOT / segment['directory']
    candidates = [p for p in directory.glob('model_*.pt') if re.fullmatch(r'model_\d+\.pt', p.name)]
    for path in sorted(candidates, key=lambda p: int(p.stem.split('_')[1]), reverse=True):
        iteration = finite_checkpoint(path)
        if iteration is not None:
            return path, iteration
    return None


def train(experiment, plan, state_path):
    state = load_json(state_path, {'experiment': experiment['name'], 'segments': []})
    if state.get('training_complete') and (ROOT / state['checkpoint']).exists():
        print(f"Обучение {experiment['name']} уже завершено; использую сохранённый результат.", flush=True)
        return state
    resume = None
    retained = 0
    start_iteration = 0
    if state['segments']:
        found = latest_finite(state['segments'][-1])
        if found is None:
            raise RuntimeError('Предыдущий запуск не завершён и корректного чекпоинта нет; логи сохранены')
        resume, start_iteration = found
        retained = retained_updates(state['segments'][-1], start_iteration)
    shutil.copyfile(ROOT / 'snapshots' / experiment['name'] / 'train.py', ROOT / 'train.py')
    while True:
        index = len(state['segments'])
        if index > plan['max_recoveries']:
            raise RuntimeError('Лимит восстановлений исчерпан; остальные опыты будут продолжены')
        suffix = '' if index == 0 else f'_resume{index}'
        directory = f"logs/{experiment['name']}{suffix}"
        destination = ROOT / directory
        if destination.exists() and any(destination.iterdir()):
            raise RuntimeError(f'Отказ перезаписывать существующие данные без состояния: {directory}')
        remaining = plan['max_iters'] - retained
        if remaining <= 0:
            raise RuntimeError('Неожиданное число сохранённых обновлений')
        segment = {'directory': directory, 'retained_before': retained,
                   'start_iteration': start_iteration, 'requested_updates': remaining,
                   'started_at': now(), 'status': 'running'}
        state['segments'].append(segment)
        save_json(state_path, state)
        command = [sys.executable, '-u', 'train.py', '--seed', str(plan['train_seed']),
                   '--num_envs', str(experiment['num_envs']), '--max_iters', str(remaining),
                   '--exp_name', experiment['name'] + suffix]
        if resume is not None:
            command += ['--resume', str(resume.relative_to(ROOT))]
        segment['command'] = command
        save_json(state_path, state)
        code, wall = stream(command, destination / 'train.log')
        text = (destination / 'train.log').read_text(encoding='utf-8')
        segment.update(parse_training(text), exit_code=code, wall_seconds=wall,
                       finished_at=now(), status='completed' if code == 0 else 'failed')
        found = latest_finite(segment)
        save_json(state_path, state)
        if code == 0 and found and segment['reported_seconds'] is not None:
            checkpoint, iteration = found
            completed = retained_updates(segment, iteration)
            if completed != plan['max_iters']:
                raise RuntimeError(f'Получено {completed} обновлений вместо {plan["max_iters"]}')
            state.update(training_complete=True, checkpoint=str(checkpoint.relative_to(ROOT)),
                         mean_reward=segment['mean_reward'], completed_updates=completed,
                         recovered=index > 0,
                         training_wall_seconds=round(sum(s.get('wall_seconds', 0) for s in state['segments']), 2),
                         training_reported_seconds=sum(s['reported_seconds'] for s in state['segments'] if s.get('reported_seconds') is not None),
                         reported_time_covers_all_segments=all(s.get('reported_seconds') is not None for s in state['segments']))
            save_json(state_path, state)
            return state
        if not found:
            raise RuntimeError(f'Обучение упало, корректного чекпоинта нет: {directory}/train.log')
        if not re.search(r'NaN values|contain NaN|non-finite|nan values', text):
            raise RuntimeError(f'Обучение упало с кодом {code}; автоматическое восстановление оставлено для численных ошибок')
        resume, start_iteration = found
        retained = retained_updates(segment, start_iteration)
        print(f'Восстановление из {resume.name}: сохранено {retained} обновлений; состояние среды будет сброшено.', flush=True)


def evaluate(checkpoint, label, plan, extra=None):
    directory = ROOT / 'evaluation' / label
    marker = directory / 'result.json'
    log_path = directory / 'inference.log'
    video = directory / 'video.mp4'
    old = load_json(marker, {})
    if old.get('status') == 'completed' and video.exists() and log_path.exists():
        return old
    attempt = 1
    if log_path.exists():
        while (directory / f'inference-attempt-{attempt}.log').exists():
            attempt += 1
        log_path.rename(directory / f'inference-attempt-{attempt}.log')
    command = [sys.executable, '-u', 'inference.py', '--checkpoint', checkpoint,
               '--seed', str(plan['eval_seed']), '--episodes', '10',
               '--record', str(video.relative_to(ROOT))] + list(extra or [])
    code, wall = stream(command, log_path)
    if code != 0:
        raise RuntimeError(f'Ошибка инференса: {log_path.relative_to(ROOT)}')
    result = parse_inference(log_path.read_text(encoding='utf-8'))
    if not video.exists() or video.stat().st_size == 0:
        raise RuntimeError('Инференс не сохранил видео')
    result.update(status='completed', seed=plan['eval_seed'], checkpoint=checkpoint,
                  extra_flags=list(extra or []), command=command,
                  wall_seconds=wall, finished_at=now(), video=str(video.relative_to(ROOT)))
    save_json(marker, result)
    return result


def preflight():
    import torch
    import jax
    from mujoco_playground import registry
    from config import build_runner_cfg
    if not torch.cuda.is_available() or not jax.devices('cuda'):
        raise RuntimeError('Требуется CUDA в PyTorch и JAX; уменьшать число сред автоматически не будем')
    scales = registry.get_default_config('PandaPickCube').reward_config.scales.to_dict()
    expected = {'gripper_box': 4.0, 'box_target': 8.0, 'no_floor_collision': 0.25, 'robot_target_qpos': 0.3}
    for key, value in expected.items():
        if scales[key] != value:
            raise RuntimeError(f'Дефолтная награда изменилась: {key}={scales[key]}')
    cfg = build_runner_cfg(66)
    if cfg['algorithm']['learning_rate'] != 0.001 or cfg['algorithm']['schedule'] != 'adaptive':
        raise RuntimeError('PPO-конфиг отличается от ожидаемого baseline')
    names = ['playground', 'rsl-rl-lib', 'torch', 'jax', 'jaxlib', 'mujoco', 'mujoco-mjx', 'numpy']
    record = {'captured_at': now(), 'python': sys.version, 'gpu': torch.cuda.get_device_name(0),
              'cuda_runtime': torch.version.cuda, 'versions': {n: importlib.metadata.version(n) for n in names},
              'default_reward_scales': scales, 'execution_note': 'Computations run on the host providing the GPU; no claim of execution on the student own hardware.'}
    save_json(ROOT / 'results' / 'runtime.json', record)
    print('Проверка GPU, библиотек и исходной конфигурации пройдена.', flush=True)


def make_charts(states):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    fig, axes = plt.subplots(2, 3, figsize=(14, 7), layout='constrained')
    for ax, state in zip(axes.flat, states):
        for segment in state.get('segments', []):
            path = ROOT / segment['directory'] / 'train.log'
            if not path.exists():
                continue
            pairs = []
            for block in re.finditer(r'Learning iteration\s+(\d+)/\d+(.*?)(?=Learning iteration|\Z)', path.read_text(encoding='utf-8'), re.S):
                match = re.search(r'Mean reward:\s*([-+\d.eE]+)', block[2])
                if match:
                    pairs.append((segment['retained_before'] + int(block[1]) - segment['start_iteration'], float(match[1])))
            if pairs:
                x, y = np.array(pairs).T
                label = Path(segment['directory']).name
                ax.plot(x, y, alpha=.3, lw=.6)
                if len(y) >= 25:
                    ax.plot(x[24:], np.convolve(y, np.ones(25) / 25, mode='valid'), label=label)
                ax.legend(fontsize=7)
        ax.set(title=state['experiment'], xlabel='Logical update index', ylabel='Training mean_reward')
        ax.grid(alpha=.2)
    axes.flat[-1].axis('off')
    axes.flat[-1].text(0, .8, 'Different reward weights change the metric scale.\nCompare task performance using SR.\nRecovery segments may overlap discarded updates.', va='top', fontsize=10)
    target = ROOT / 'results' / 'learning-curves.png'
    fig.savefig(target, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry-run', action='store_true', help='Show plan without importing ML libraries or running anything')
    parser.add_argument('--check', action='store_true', help='Check installed environment; do not train')
    args = parser.parse_args()
    os.chdir(ROOT)
    plan = load_json(ROOT / 'plan.json', None)
    if not plan:
        raise RuntimeError('plan.json отсутствует')
    if args.dry_run:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return 0
    import fcntl
    with (ROOT / '.batch.lock').open('w') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('Эта серия уже запущена; второй параллельный процесс запрещён')
        preflight()
        if args.check:
            return 0
        digest = hashlib.sha256((ROOT / 'plan.json').read_bytes()).hexdigest()
        session_path = ROOT / 'results' / 'session.json'
        session = load_json(session_path, {'plan_sha256': digest, 'first_started_at': now()})
        if session['plan_sha256'] != digest:
            raise RuntimeError('План изменён после начала серии. Для другого плана нужна новая папка.')
        save_json(session_path, session)
        failures = []
        states = []
        for index, experiment in enumerate(plan['experiments'], 1):
            print(f'\n{"=" * 70}\nОПЫТ {index}/{len(plan["experiments"])}: {experiment["name"]}\n{"=" * 70}', flush=True)
            state_path = ROOT / 'state' / (experiment['name'] + '.json')
            try:
                state = train(experiment, plan, state_path)
                state['evaluation'] = evaluate(state['checkpoint'], experiment['name'], plan)
                state['status'] = 'completed'
                state.pop('last_error', None)
                save_json(state_path, state)
            except Exception as error:
                print(f'ОШИБКА {experiment["name"]}: {error}\nПерехожу к следующему опыту.', flush=True)
                state = load_json(state_path, {'experiment': experiment['name']})
                state.update(status='failed', last_error=str(error))
                save_json(state_path, state)
                failures.append(experiment['name'])
            states.append(state)
            save_json(ROOT / 'results' / 'experiments.json', states)
        baseline = next((s for s in states if s['experiment'] == 'baseline'), None)
        robustness = []
        if baseline and baseline.get('training_complete'):
            for case in plan['robustness_cases']:
                print(f'\nПРОВЕРКА УСТОЙЧИВОСТИ: {case["name"]}', flush=True)
                try:
                    result = evaluate(baseline['checkpoint'], 'robustness/' + case['name'], plan, case['flags'])
                    robustness.append(dict(case, **result))
                except Exception as error:
                    robustness.append(dict(case, status='failed', error=str(error)))
                    failures.append('robustness/' + case['name'])
                save_json(ROOT / 'results' / 'robustness.json', robustness)
        else:
            failures.append('robustness/baseline_unavailable')
        try:
            make_charts(states)
        except Exception as error:
            print(f'График не построен: {error}', flush=True)
            failures.append('learning_curves')
        summary = {'finished_at': now(), 'status': 'completed' if not failures else 'completed_with_errors',
                   'failures': failures, 'experiments': len(states), 'robustness_cases': len(robustness)}
        save_json(ROOT / 'results' / 'summary.json', summary)
        print('\nСЕРИЯ ЗАВЕРШЕНА\n' + json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
        print('Логи, модели, видео и машинные результаты сохранены в этой папке. Отчёты заполняются после анализа.', flush=True)
        return 0 if not failures else 1


if __name__ == '__main__':
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print('\nСерия остановлена пользователем. Состояния и логи сохранены; та же команда продолжит незавершённые этапы.', flush=True)
        sys.exit(130)
