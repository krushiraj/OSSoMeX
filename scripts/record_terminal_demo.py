"""Record actual interactive CLI output as asciicast, text and a terminal GIF.

Optional tools: pillow, pyte, pexpect. Run with --help for paths.
"""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path
import pexpect
import pyte
from PIL import Image, ImageDraw, ImageFont

root = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--python', default='.venv-scibert/bin/python', help='Installed model environment')
parser.add_argument('--model', default='model-assets/checkpoints/scibert-full-label-006')
parser.add_argument('--device', choices=('cpu', 'mps'), default='cpu')
parser.add_argument('--font', required=True, help='Path to a monospace TrueType font')
parser.add_argument('--output', default='docs/assets', help='Recording directory; replaces terminal-demo files')
options = parser.parse_args()
model = root / options.model
out = root / options.output
out.mkdir(parents=True, exist_ok=True)
cols, rows = 112, 16
samples = [
    'We used NumPy 1.24.3 for numerical analysis.',
    'The simulations were performed in MATLAB R2023b.',
    'We used Python 3.11 and pandas 2.0.3 to process the data.',
    'The samples were stored at room temperature.',
]
args = ['-m', 'research', 'full-label', 'interactive', '--model',
        options.model, '--device', options.device, '--format', 'table']
command = options.python + ' ' + ' '.join(args)
start = time.monotonic()
events = []
screen = pyte.Screen(cols, rows)
stream = pyte.Stream(screen)
font = ImageFont.truetype(options.font, 16)
small = ImageFont.truetype(options.font, 13)
cw, ch = 10, 23
width, height = cols*cw+40, rows*ch+76
frames, frame_times = [], []
last_frame = 0
palette = {'black':'#1b2030', 'red':'#f38ba8', 'green':'#a6e3a1', 'brown':'#f9e2af',
 'blue':'#89b4fa', 'magenta':'#cba6f7', 'cyan':'#94e2d5', 'white':'#cdd6f4', 'default':'#d8dfeb',
 'brightblack':'#8994a8', 'brightred':'#f38ba8', 'brightgreen':'#a6e3a1', 'brightyellow':'#f9e2af',
 'brightblue':'#89b4fa', 'brightmagenta':'#cba6f7', 'brightcyan':'#94e2d5', 'brightwhite':'#ffffff'}

def frame(t, force=False):
    global last_frame
    if not force and t-last_frame < .15:
        return
    im = Image.new('RGB', (width,height), '#111827')
    d = ImageDraw.Draw(im)
    d.rectangle((0,0,width,43), fill='#202b3d')
    for i,c in enumerate(['#ff615a','#ffbd2e','#28c840']):
        d.ellipse((16+i*23,16,27+i*23,27),fill=c)
    d.text((110,12),f'OSSoMeX | {model.name} | {options.device.upper()} | recorded CLI',font=small,fill='#e2e8f0')
    for y in range(rows):
        for x, cell in screen.buffer[y].items():
            if not cell.data.strip():
                continue
            fg = palette.get(cell.fg, '#' + cell.fg if len(cell.fg)==6 else '#d8dfeb')
            d.text((20+x*cw,54+y*ch),cell.data,font=font,fill=fg)
    frames.append(im)
    frame_times.append(t)
    last_frame=t

class Recorder:
    def write(self, data):
        t=time.monotonic()-start
        events.append([round(t,6),'o',data])
        stream.feed(data)
        frame(t)
    def flush(self): pass

rec = Recorder()
rec.write('$ ' + command + '\r\n')
env = dict(os.environ, TERM='xterm-256color', PROMPT_TOOLKIT_NO_CPR='1', TOKENIZERS_PARALLELISM='false')
child=pexpect.spawn(str(root / options.python),args,cwd=str(root),env=env,
                    encoding='utf-8',dimensions=(rows,cols),timeout=120)
child.logfile_read=rec
transcripts=[]
try:
    child.expect_exact('text>')
    for i,text in enumerate(samples,1):
        child.send('\x1b[200~'+text+'\x1b[201~')
        child.expect_exact(text)
        time.sleep(.6)
        child.send('\r')
        child.expect_exact(f'interactive-{i}:')
        child.expect_exact('text>')
        frame(time.monotonic()-start,True)
        snapshot='\n'.join(line.rstrip() for line in screen.display).strip()
        transcripts.append(snapshot)
        print(f'Captured sample {i}:\n{snapshot}\n',flush=True)
        time.sleep(6)
        if i < len(samples):
            child.send('\x0c')
            child.expect_exact('text>')
    child.send('\x04')
    child.expect(pexpect.EOF)
    child.close()
    if child.exitstatus != 0:
        raise RuntimeError(f'CLI exit {child.exitstatus}')
finally:
    if child.isalive(): child.terminate(force=True)
frame(time.monotonic()-start,True)
header={'version':2,'width':cols,'height':rows,'timestamp':int(time.time()),'env':{'TERM':'xterm-256color'},'title':f'{model.name}: four real interactive samples','command':command}
(out/'terminal-demo.cast').write_text('\n'.join(json.dumps(v) for v in [header,*events])+'\n')
(out/'terminal-demo.txt').write_text('OSSoMeX interactive demo\nCommand: '+command+'\nSynthetic examples; actual model outputs, not benchmark gold.\n\n'+'\n\n---\n\n'.join(transcripts)+'\n')
durations=[max(80,min(6000,int((b-a)*1000))) for a,b in zip(frame_times,frame_times[1:])]+[3000]
frames[0].save(out/'terminal-demo.gif',save_all=True,append_images=frames[1:],duration=durations,loop=0,optimize=True)
(out/'terminal-demo.json').write_text(json.dumps({'checkpoint':model.name,'device':options.device,'command':command,'samples':samples,'exit_status':child.exitstatus,'recording':'Real PTY output rendered as a terminal GIF; idle gaps capped at 6 seconds. Not a speed benchmark.','manifest_sha256':hashlib.sha256((model/'manifest.json').read_bytes()).hexdigest()},indent=2)+'\n')
print(f'Saved {len(frames)} frames, {sum(durations)/1000:.1f}s; GIF {(out/"terminal-demo.gif").stat().st_size} bytes')
