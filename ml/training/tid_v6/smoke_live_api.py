"""Exercise the running local websocket with generated fixtures, not a camera."""
import asyncio
import json
from pathlib import Path
import sys

import numpy as np
import websockets

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'apps/api'))


def hands(points):
    return [dict(landmarks=np.column_stack([np.array(hand) * .1 + [.3, .4], np.zeros(21)]).tolist(),
                 handedness=['Right', 'Left'][i], handedness_score=.99)
            for i, hand in enumerate(points)]


async def main():
    sequences = json.loads((Path(__file__).resolve().parents[3] /
        'artifacts/tid-v6/motion/authored_sequences.json').read_text(encoding='utf-8'))['sequences']
    report = {'scope': 'generated geometry / running websocket integration, not webcam accuracy'}
    async with websockets.connect('ws://127.0.0.1:8006/ws/recognize', origin='http://localhost:3000') as socket:
        hello = json.loads(await socket.recv())
        assert hello['recognition_revision'] == 'g-n-1', hello
        report['recognition_revision'] = hello['recognition_revision']
        for letter in ('N', 'Ğ'):
            await socket.send(json.dumps({'action': 'clear'}))
            await socket.recv()
            rows = sequences[letter]
            if letter == 'N':
                rows = [dict(time=i / 10, points=rows[0]['points']) for i in range(22)]
            accepted, latency = [], []
            for row in rows:
                observed = hands(row['points'])
                await socket.send(json.dumps(dict(hands=observed, aspect_ratio=1,
                    motion_frames=[dict(timestamp_ms=row['time'] * 1000, hands=observed)])))
                result = json.loads(await socket.recv())
                assert not result.get('error'), result
                latency.append(result.get('latency_ms', 0))
                if result.get('ready'):
                    accepted.append((result['prediction'], result['source']))
                if letter == 'N':
                    await asyncio.sleep(.1)  # N's stability uses server wall time.
            expected_source = 'tid_n_pose_rule' if letter == 'N' else 'tid_g_index_bob_rule'
            assert (letter, expected_source) in accepted, (letter, accepted)
            report[letter] = dict(accepted=True, source=expected_source,
                                  median_latency_ms=round(float(np.median(latency)), 2))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    asyncio.run(main())
