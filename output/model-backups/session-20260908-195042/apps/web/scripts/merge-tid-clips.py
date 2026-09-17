"""Combine simultaneous left/right glTF tracks into one portable clip."""
import json
import struct
from pathlib import Path

def merge(path):
    raw = path.read_bytes()
    magic, version, length = struct.unpack_from('<III', raw)
    assert magic == 0x46546C67 and version == 2 and length == len(raw)
    size, kind = struct.unpack_from('<II', raw, 12)
    assert kind == 0x4E4F534A
    doc = json.loads(raw[20:20 + size])
    merged = {'name': 'TID_Alphabet', 'samplers': [], 'channels': []}
    targets = set()
    for clip in doc['animations']:
        offset = len(merged['samplers'])
        merged['samplers'].extend(clip['samplers'])
        for channel in clip['channels']:
            target = (channel['target']['node'], channel['target']['path'])
            assert target not in targets, target
            targets.add(target)
            merged['channels'].append({**channel, 'sampler': channel['sampler'] + offset})
    doc['animations'] = [merged]
    encoded = json.dumps(doc, separators=(',', ':'), ensure_ascii=False).encode('utf-8')
    encoded += b' ' * (-len(encoded) % 4)
    remainder = raw[20 + size:]
    output = struct.pack('<III', magic, version, 20 + len(encoded) + len(remainder))
    output += struct.pack('<II', len(encoded), kind) + encoded + remainder
    path.write_bytes(output)
    print(f'Merged {len(targets)} tracks into TID_Alphabet: {path}')

if __name__ == '__main__':
    merge(Path(__file__).resolve().parents[3] / 'output/models/tid-alphabet-animated.glb')
