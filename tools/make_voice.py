#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""สร้างชุดเสียงพากย์ (voice pack) ของเกมดีดลูกแก้ว ด้วย Gemini TTS

  1. อ่านบทพูดจาก index.html (ตาราง VOICE_LINES ระหว่าง VOICE_LINES_BEGIN / VOICE_LINES_END)
  2. เรียก Gemini TTS ทีละประโยค พร้อมกำกับอารมณ์ของประโยคนั้น
  3. ตัดความเงียบหัวท้าย ปรับความดังให้เท่ากัน แปลงเป็น mp3 เก็บใน voice/
  4. เขียน voice/pack.json ให้เกมอ่าน (เกมจะเล่นคลิปแทนเสียงอ่านของเครื่อง)

ทำต่อจากของเดิมได้: ประโยคที่มีคลิปแล้วจะข้าม ถ้าโควตารายวันหมดจะบันทึกเท่าที่ได้แล้วจบ รันใหม่ภายหลังเพื่อทำต่อ
ใช้แค่ Python มาตรฐาน + ffmpeg  คีย์ API อ่านจากตัวแปร GEMINI_API_KEY เท่านั้น (ห้ามใส่คีย์ลงไฟล์)

ตัวอย่าง:
  GEMINI_API_KEY=... python3 tools/make_voice.py --voice Leda
  python3 tools/make_voice.py --dry-run          # ดูรายการประโยคทั้งหมด ไม่เรียก API
"""
import argparse
import base64
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request

DEFAULT_VOICE = 'Leda'
DEFAULT_MODEL = 'gemini-3.8-flash-tts'          # รุ่นที่รองรับภาษาไทย
API_BASE = 'https://generativelanguage.googleapis.com/v1beta'


class DailyQuota(Exception):
    pass


class Fatal(Exception):
    pass


class Retryable(Exception):
    pass


def log(*a):
    print(*a, flush=True)


# ----------------------------------------------------------------------------- บทพูด
def load_lines(index_path):
    html = open(index_path, encoding='utf-8').read()
    m = re.search(r'/\*VOICE_LINES_BEGIN\*/(.*?)/\*VOICE_LINES_END\*/', html, re.S)
    if not m:
        raise Fatal('ไม่พบตาราง VOICE_LINES ใน ' + index_path)
    block = m.group(1)
    return json.loads(block[block.index('{'):block.rindex('}') + 1])


def expand(line, bots):
    """แทน {P} / {S} {V} ด้วยชื่อบอททุกแบบที่เกิดขึ้นได้ในเกม"""
    if '{S}' in line or '{V}' in line:
        return [line.replace('{S}', s).replace('{V}', v) for s in bots for v in bots if s != v]
    if '{P}' in line:
        return [line.replace('{P}', b) for b in bots]
    return [line]


def build_jobs(L):
    """เรียงงานแบบวนรอบ: ประโยคแรกของทุกเหตุการณ์ก่อน แล้วค่อยประโยคที่สอง...
    ถ้าโควตาหมดกลางทาง เหตุการณ์สำคัญ (ชนะ ลงหลุม พลาด) จะมีเสียงก่อน"""
    persona = L.get('persona', '').strip()
    per_pool = []
    for key, pool in L['pools'].items():
        style = L['moods'][pool['mood']]['style'].strip()
        if persona:
            style = style + '; ' + persona
        texts = [t for line in pool['lines'] for t in expand(line, L.get('bots', []))]
        per_pool.append([{'pool': key, 'text': t, 'style': style} for t in texts])
    jobs, seen, r = [], set(), 0
    while any(r < len(p) for p in per_pool):
        for p in per_pool:
            if r < len(p) and p[r]['text'] not in seen:
                seen.add(p[r]['text'])
                jobs.append(p[r])
        r += 1
    return jobs


def clip_name(voice, job):
    h = hashlib.sha1('|'.join([voice, job['style'], job['text']]).encode('utf-8')).hexdigest()[:10]
    return '%s-%s.mp3' % (re.sub(r'[^a-z0-9]', '', voice.lower()) or 'voice', h)


# ----------------------------------------------------------------------------- เรียก API
def call_tts(api_base, key, model, voice, text, style, timeout=120):
    body = {
        'contents': [{'role': 'user', 'parts': [{'text': text, 'speech_metadata': {'style': style}}]}],
        'generationConfig': {
            'responseModalities': ['AUDIO'],
            'speechConfig': {'voiceConfig': {'voice': voice}},
        },
    }
    req = urllib.request.Request(
        '%s/models/%s:generateContent' % (api_base.rstrip('/'), model),
        data=json.dumps(body, ensure_ascii=False).encode('utf-8'),
        headers={'x-goog-api-key': key, 'Content-Type': 'application/json'},
        method='POST')
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            resp = json.loads(r.read().decode('utf-8'))
    except urllib.error.HTTPError as e:
        raw = e.read().decode('utf-8', 'replace')
        try:
            err = json.loads(raw).get('error', {})
        except ValueError:
            err = {}
        msg = err.get('message') or raw[:300]
        if e.code == 429:
            blob = json.dumps(err).lower()
            if 'perday' in blob or 'per_day' in blob or 'per day' in blob or 'daily' in blob:
                raise DailyQuota(msg)
            delay = 30.0
            m = re.search(r'"retrydelay":\s*"([\d.]+)s"', blob)
            if m:
                delay = float(m.group(1)) + 1
            ex = Retryable('ถี่เกินโควตาต่อนาที: ' + msg[:160])
            ex.delay = min(delay, 90)
            raise ex
        if e.code in (500, 502, 503, 504):
            ex = Retryable('เซิร์ฟเวอร์ไม่ว่าง (%d): %s' % (e.code, msg[:160]))
            ex.delay = 15
            raise ex
        raise Fatal('API ตอบ %d: %s' % (e.code, msg))
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        ex = Retryable('เชื่อมต่อไม่ได้: %s' % e)
        ex.delay = 10
        raise ex

    for cand in resp.get('candidates') or []:
        for part in (cand.get('content') or {}).get('parts') or []:
            inline = part.get('inlineData') or part.get('inline_data')
            if inline and inline.get('data'):
                return base64.b64decode(inline['data']), (inline.get('mimeType') or inline.get('mime_type') or '')
    reason = ''
    if resp.get('candidates'):
        reason = str(resp['candidates'][0].get('finishReason', ''))
    if resp.get('promptFeedback'):
        reason += ' ' + json.dumps(resp['promptFeedback'], ensure_ascii=False)[:160]
    ex = Retryable('ไม่มีเสียงในคำตอบ %s' % reason.strip())
    ex.delay = 5
    raise ex


# ----------------------------------------------------------------------------- แปลงไฟล์เสียง
def ffmpeg(args):
    return subprocess.run(['ffmpeg', '-hide_banner', '-nostdin'] + args, capture_output=True, text=True, errors='replace')


def to_mp3(raw, mime, out_path):
    """ตัดความเงียบหัวท้าย → ปรับยอดเสียงเป็น -1.5 dBFS → mp3 โมโน 44.1 kHz  คืนความยาว (วินาที)"""
    with tempfile.TemporaryDirectory() as td:
        if raw[:4] == b'RIFF':
            src = os.path.join(td, 'in.wav')
            in_args = ['-i', src]
        else:   # PCM ดิบ 16 บิต (รุ่นเก่า/สตรีม)
            m = re.search(r'rate=(\d+)', mime or '')
            src = os.path.join(td, 'in.pcm')
            in_args = ['-f', 's16le', '-ar', m.group(1) if m else '24000', '-ac', '1', '-i', src]
        with open(src, 'wb') as f:
            f.write(raw)
        trim = ('silenceremove=start_periods=1:start_threshold=-45dB:start_silence=0.03,areverse,'
                'silenceremove=start_periods=1:start_threshold=-45dB:start_silence=0.08,areverse')
        p = ffmpeg(in_args + ['-af', trim + ',volumedetect', '-f', 'null', '-'])
        m = re.search(r'max_volume:\s*(-?[\d.]+) dB', p.stderr)
        if p.returncode != 0 or not m:
            raise Retryable('ffmpeg อ่านเสียงไม่ได้: ' + p.stderr[-200:])
        gain = max(-12.0, min(24.0, -1.5 - float(m.group(1))))
        tmp = os.path.join(td, 'out.mp3')
        p = ffmpeg(in_args + ['-y', '-af', '%s,volume=%.2fdB' % (trim, gain), '-ar', '44100', '-ac', '1',
                              '-c:a', 'libmp3lame', '-b:a', '64k', tmp])
        if p.returncode != 0:
            raise Fatal('ffmpeg แปลง mp3 ไม่สำเร็จ: ' + p.stderr[-300:])
        p = ffmpeg(['-i', tmp, '-f', 'null', '-'])
        times = re.findall(r'time=(\d+):(\d+):([\d.]+)', p.stderr)
        dur = (int(times[-1][0]) * 3600 + int(times[-1][1]) * 60 + float(times[-1][2])) if times else 0.0
        os.replace(tmp, out_path)
        return dur


def max_len(text):
    """ความยาวคลิปที่ยอมรับได้ (กันกรณีโมเดลพูดเกินบท)"""
    return 2.0 + 0.25 * len(re.sub(r'[\s!?.,]', '', text))


# ----------------------------------------------------------------------------- main
def write_pack(path, pack):
    tmp = path + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(pack, f, ensure_ascii=False, indent=1)
        f.write('\n')
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser(description='สร้างชุดเสียงพากย์ด้วย Gemini TTS')
    ap.add_argument('--index', default='index.html')
    ap.add_argument('--out', default='voice')
    ap.add_argument('--voice', default='', help='ชื่อเสียง เช่น Leda, Puck, Fenrir (ว่าง = ใช้เสียงเดิมของชุดที่มีอยู่)')
    ap.add_argument('--model', default=DEFAULT_MODEL)
    ap.add_argument('--redo', action='store_true', help='สร้างใหม่ทั้งหมด ทับของเดิม')
    ap.add_argument('--max', type=int, default=0, help='จำกัดจำนวนคลิปที่สร้างรอบนี้ (0 = ไม่จำกัด)')
    ap.add_argument('--rpm', type=float, default=3.0, help='จำนวนคำขอต่อนาที (โควตาฟรี = 3)')
    ap.add_argument('--api-base', default=API_BASE)
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()

    L = load_lines(a.index)
    jobs = build_jobs(L)
    pack_path = os.path.join(a.out, 'pack.json')
    old = {}
    if os.path.exists(pack_path):
        try:
            old = json.load(open(pack_path, encoding='utf-8'))
        except ValueError:
            old = {}
    voice = (a.voice or '').strip() or old.get('voice') or DEFAULT_VOICE
    for j in jobs:
        j['file'] = clip_name(voice, j)

    if a.dry_run:
        for i, j in enumerate(jobs, 1):
            log('%2d  %-12s %s   [%s]' % (i, j['pool'], j['text'], j['style'].split(';')[0]))
        log('รวม %d ประโยค · เสียง %s · รุ่น %s' % (len(jobs), voice, a.model))
        return 0

    os.makedirs(a.out, exist_ok=True)
    wanted = {j['file'] for j in jobs}
    clips = {}
    if not a.redo and old.get('voice') == voice:
        for j in jobs:   # เก็บเฉพาะคลิปที่ยังตรงกับบท/อารมณ์/เสียงปัจจุบัน และไฟล์ยังอยู่จริง
            if old.get('clips', {}).get(j['text']) == j['file'] and os.path.exists(os.path.join(a.out, j['file'])):
                clips[j['text']] = j['file']
    todo = [j for j in jobs if j['text'] not in clips]
    reset = bool(old.get('clips')) and (a.redo or old.get('voice') != voice)   # เริ่มชุดใหม่ (เปลี่ยนเสียง/สร้างใหม่ทั้งหมด)
    made = 0
    pack = {'voice': voice, 'model': a.model, 'rev': old.get('rev', 0), 'total': len(jobs), 'clips': clips}

    def cleanup():   # ลบไฟล์คลิปที่ไม่ได้ใช้แล้ว (เปลี่ยนเสียง/แก้บท)
        keep = set(pack['clips'].values())
        for fn in os.listdir(a.out):
            if fn.endswith('.mp3') and fn not in keep:
                os.remove(os.path.join(a.out, fn))

    summary = []
    status = 'done'
    key = os.environ.get('GEMINI_API_KEY', '').strip()
    if not todo:
        log('ครบแล้ว %d/%d ประโยค (เสียง %s) ไม่มีอะไรต้องทำ' % (len(clips), len(jobs), voice))
    elif not key:
        status = 'nokey'
        log('ยังไม่ได้ตั้งค่า GEMINI_API_KEY (Settings → Secrets and variables → Actions → New repository secret)')
    else:
        log('เสียง %s · รุ่น %s · ต้องสร้าง %d จาก %d ประโยค' % (voice, a.model, len(todo), len(jobs)))
        gap = 60.0 / max(a.rpm, 0.1) + (1.0 if a.rpm <= 60 else 0.0)
        last = 0.0
        try:
            for j in todo:
                if a.max and made >= a.max:
                    status = 'max'
                    break
                dur = None
                for attempt in range(1, 6):
                    wait = last + gap - time.time()
                    if wait > 0:
                        time.sleep(wait)
                    last = time.time()
                    try:
                        raw, mime = call_tts(a.api_base, key, a.model, voice, j['text'], j['style'])
                        dur = to_mp3(raw, mime, os.path.join(a.out, j['file']))
                        if dur < 0.2:
                            raise Retryable('คลิปว่าง')
                        if dur > max_len(j['text']):
                            raise Retryable('คลิปยาวผิดปกติ %.1f วิ (โมเดลอาจพูดเกินบท)' % dur)
                        break
                    except Retryable as e:
                        dur = None
                        log('   … %s (ครั้งที่ %d) %s' % (j['text'], attempt, e))
                        if os.path.exists(os.path.join(a.out, j['file'])):
                            os.remove(os.path.join(a.out, j['file']))
                        rate = 'โควตาต่อนาที' in str(e)
                        if attempt >= (5 if rate else 3):
                            if rate:   # ติดโควตาไม่เลิก = ถือว่าหมดโควตา หยุดอย่างสุภาพ
                                raise DailyQuota(str(e))
                            break
                        time.sleep(getattr(e, 'delay', 5))
                if dur is None:
                    summary.append((j, None))
                    continue
                pack['clips'][j['text']] = j['file']
                pack['rev'] = int(time.time())
                write_pack(pack_path, pack)   # บันทึกทุกคลิป เผื่อหยุดกลางทาง
                made += 1
                summary.append((j, dur))
                log('✓ %2d/%d  %.1f วิ  %s' % (len(pack['clips']), len(jobs), dur, j['text']))
        except DailyQuota as e:
            status = 'quota'
            log('โควตารายวันหมด: %s' % str(e)[:200])
        except Fatal as e:
            status = 'fatal'
            log('ผิดพลาด: %s' % e)

    # จัดลำดับ clips ตามลำดับบท แล้วเขียนไฟล์/เก็บกวาด
    pack['clips'] = {j['text']: pack['clips'][j['text']] for j in jobs if j['text'] in pack['clips']}
    if made == 0 and (reset or status in ('nokey', 'fatal')):
        # ยังสร้างไม่ได้สักคลิป → คงของเดิมไว้ทั้งหมด ไม่เขียน ไม่ลบอะไร
        pack = dict(old) if old else pack
    elif made or pack['clips'] != old.get('clips') or len(jobs) != old.get('total'):
        pack['rev'] = int(time.time())
        write_pack(pack_path, pack)
        cleanup()

    n, total = len(pack.get('clips', {})), len(jobs)
    if reset and made == 0 and status == 'done':
        status = 'fatal'
    head = {
        'done': '✅ เสียงพากย์ครบ %d/%d ประโยค' % (n, total) if n == total else '⚠️ ได้ %d/%d ประโยค (บางประโยคสร้างไม่สำเร็จ รันอีกครั้งเพื่อลองใหม่)' % (n, total),
        'max': '⏸️ ได้ %d/%d ประโยค (ครบจำนวนที่กำหนดรอบนี้)' % (n, total),
        'quota': '⏳ ได้ %d/%d ประโยค โควตาฟรีของวันนี้หมด ระบบจะทำต่อเองวันพรุ่งนี้ (หรือกด Run workflow อีกครั้งภายหลัง)' % (n, total),
        'nokey': '🔑 ยังไม่ได้ตั้งค่า GEMINI_API_KEY ใน Settings → Secrets and variables → Actions',
        'fatal': '❌ หยุดเพราะข้อผิดพลาด (ดู log) — ได้ %d/%d ประโยค' % (n, total),
    }[status]
    lines = ['## ' + head, '', 'เสียง: **%s** · รุ่น: `%s`' % (voice, a.model), '']
    if summary:
        lines += ['| ประโยค | อารมณ์ | ความยาว |', '|---|---|---|']
        for s in summary:
            j = s[0]
            lines.append('| %s | %s | %s |' % (j['text'], j['style'].split(';')[0], 'ไม่สำเร็จ' if s[1] is None else '%.1f วิ' % s[1]))
    log('\n' + head)
    if os.environ.get('GITHUB_STEP_SUMMARY'):
        with open(os.environ['GITHUB_STEP_SUMMARY'], 'a', encoding='utf-8') as f:
            f.write('\n'.join(lines) + '\n')
    if status == 'fatal':
        return 1
    if status == 'nokey' and os.environ.get('GITHUB_EVENT_NAME') != 'schedule':
        return 1
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except Fatal as e:
        log('ผิดพลาด: %s' % e)
        sys.exit(1)
