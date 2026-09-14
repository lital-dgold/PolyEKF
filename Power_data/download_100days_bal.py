import bz2
import csv
import json
import os
import threading
import time
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta

from extract_full_graph import extract_full_graph

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = 'https://huggingface.co/datasets/OpenSynth/D-GITT-RTE7000-2021/resolve/main'
OUT_CSV = os.path.join(HERE, 'rte7000_data', 'bal100_timeseries.csv')
LOG_PATH = os.path.join(HERE, 'rte7000_data', 'download_100days_bal_log.txt')

with open(os.path.join(HERE, 'rte7000_data', 'branch_params_bal.json')) as f:
    branch_params = json.load(f)
BRANCH_IDS = [b['id'] for b in branch_params]  # 130 branch ids, fixed column order

START = datetime(2021, 6, 1, 0, 0)
N_DAYS = 100
N_STEPS_PER_DAY = 288

log_lock = threading.Lock()
def log(msg):
    ts = datetime.now().strftime('%H:%M:%S')
    line = f'[{ts}] {msg}'
    print(line, flush=True)
    with log_lock:
        with open(LOG_PATH, 'a') as f:
            f.write(line + '\n')

counters = {'ok': 0, 'missing_404': 0, 'failed': 0}
counters_lock = threading.Lock()
file_lock = threading.Lock()

def fetch_and_process(dt, path):
    url = f'{BASE}/{path}'
    data = None
    for attempt in range(6):
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=30) as r:
                data = r.read()
            break
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return dt, None, 'missing'
            time.sleep(1.5)
        except Exception:
            time.sleep(1.5)
    if data is None:
        return dt, None, 'failed'
    try:
        raw = bz2.decompress(data)
        _, edges, _ = extract_full_graph(raw)
        edge_by_id = {e['id']: e for e in edges}
        row = []
        for bid in BRANCH_IDS:
            e = edge_by_id.get(bid)
            if e is None:
                row.append('')
            else:
                connected = bool(e['connected1'] and e['connected2'] and e['bus1'] and e['bus2'])
                row.append(1 if connected else 0)
        return dt, row, 'ok'
    except Exception as ex:
        return dt, None, f'parse_error:{ex!r}'

def main():
    candidates = []
    for d in range(N_DAYS):
        day = START + timedelta(days=d)
        for step in range(N_STEPS_PER_DAY):
            dt = day + timedelta(minutes=5 * step)
            fname = f'recollement-auto-{dt.strftime("%Y%m%d")}-{dt.strftime("%H%M")}-enrichi.xiidm.bz2'
            path = f'{dt.strftime("%Y")}/{dt.strftime("%m")}/{dt.strftime("%d")}/{fname}'
            candidates.append((dt, path))

    log(f'Total candidate snapshots: {len(candidates)} across {N_DAYS} days from {START.date()}')

    already_done = set()
    write_header = True
    if os.path.exists(OUT_CSV):
        with open(OUT_CSV, 'r', newline='') as f:
            reader = csv.reader(f)
            header = next(reader, None)
            if header:
                write_header = False
            for row in reader:
                if row:
                    already_done.add(row[0])
        log(f'Resuming: {len(already_done)} timestamps already in {OUT_CSV}')

    todo = [(dt, path) for dt, path in candidates if dt.isoformat() not in already_done]
    log(f'Remaining to fetch: {len(todo)}')

    csv_file = open(OUT_CSV, 'a', newline='')
    csv_writer = csv.writer(csv_file)
    if write_header:
        csv_writer.writerow(['timestamp'] + BRANCH_IDS)
        csv_file.flush()

    MAX_WORKERS = 6
    t_start = time.time()
    n_done = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(fetch_and_process, dt, path): (dt, path) for dt, path in todo}
        for fut in as_completed(futures):
            dt, path = futures[fut]
            result_dt, row, status = fut.result()
            n_done += 1
            with counters_lock:
                if status == 'ok':
                    counters['ok'] += 1
                elif status == 'missing':
                    counters['missing_404'] += 1
                else:
                    counters['failed'] += 1
            if row is not None:
                with file_lock:
                    csv_writer.writerow([result_dt.isoformat()] + row)
                    csv_file.flush()
            if n_done % 200 == 0 or n_done == len(todo):
                elapsed = time.time() - t_start
                rate = n_done / elapsed if elapsed > 0 else 0
                eta_min = (len(todo) - n_done) / rate / 60 if rate > 0 else float('inf')
                log(f'progress: {n_done}/{len(todo)}  ok={counters["ok"]} missing={counters["missing_404"]} failed={counters["failed"]}  rate={rate:.2f}/s  ETA={eta_min:.1f}min')

    csv_file.close()
    log(f'DONE. ok={counters["ok"]} missing={counters["missing_404"]} failed={counters["failed"]}')

if __name__ == '__main__':
    main()
