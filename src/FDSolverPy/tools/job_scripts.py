import os, subprocess

def shell_to_strlist(command):
    err,output = subprocess.getstatusoutput(command)
    if err == 0: return output.strip().split('\n')
    else: return []
    
def cpus_per_node(p='pdebug'):
    err, output = subprocess.getstatusoutput('sinfo --Node --long -p %s | grep %s' % (p, p))
    if err == 0:
        return int(output.strip().split('\n')[(-1)].split()[4])
    return 0


def number_of_nodes(n, ncpus_per_node=None, p='pdebug'):
    if ncpus_per_node is None:
        ncpus_per_node = cpus_per_node(p)
    N = int((n - (n - 1) % ncpus_per_node - 1) / ncpus_per_node) + 1
    return N


def parse_time(ss, t=60):
    time_list = list(map(float, re.split('[hms]', ss)[:-1]))
    time = sum(map(lambda x: x[1] * t ** x[0], enumerate(reversed(time_list))))
    return time


def write_jobscript(path, job_script, alloc, env, run):
    f = open(os.path.join(path, job_script), 'wt')
    f.write('#!/bin/bash\n')
    f.write(''.join([f"#SBATCH -{key} {value}\n" for key, value in alloc.items()]))
    f.write("echo -n 'JobID is '; echo $SLURM_JOB_ID\n\n")
    f.write('#### Setup environment\n')
    f.write('\n'.join(env))
    f.write('#### RUN\n')
    f.write(f"time {run}\n")
    f.close()


def submit_job(path, job_script='submit.sh', alloc={}, env={}, run='srun run.py'):
    write_jobscript(path, job_script, alloc, env, run)
    cwd = os.getcwd()
    os.chdir(path)
    subprocess.call(('sbatch %s' % job_script), shell=True)
    os.chdir(cwd)
