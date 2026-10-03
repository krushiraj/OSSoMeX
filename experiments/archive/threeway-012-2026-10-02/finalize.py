import os,subprocess,sys,time
from run import ROOT,OUT
while 'COMPLETE WHITESPACE CONTROL' not in (OUT/'whitespace-control.log').read_text():
 if 'Traceback (most recent call last)' in (OUT/'whitespace-control.log').read_text():raise RuntimeError('Whitespace control failed')
 time.sleep(5)
commands=[['timing_cpu.py'],['score.py'],['score.py',str(OUT/'corrected')],['score.py',str(OUT/'whitespace-control')],['score_whitespace_canonical.py'],['surface_diagnostic.py'],['audit_training_labels.py'],['diagnostics.py'],['alias_diagnostic.py'],['build_error_viewer.py'],['build_examples.py'],['plot_results.py'],['build_report.py'],['check_harness.py'],['verify_run.py']]
for command in commands:
 print('FINALIZE',command,flush=True)
 result=subprocess.run(['python3' if command[0]=='plot_results.py' else sys.executable,str(OUT/command[0]),*command[1:]],cwd=ROOT)
 if result.returncode:raise RuntimeError('Finalization failed: '+str(command))
print('COMPLETE FINALIZATION',flush=True)
