#* Parallel and Distributed Programming Hands-on Environment

<!--- md w --->

Enter your name and student ID.

 * Name:
 * Student ID:

<!--- end md --->

# How this page works (Jupyter notebook basics)

* This page is a Jupyter notebook
* It consists of text cells and code cells

## Cell

* A text box like the one below is called a "cell"
* Press SHIFT+ENTER to execute it

## Python code cell

<!--- code w kernel=python --->
def f(x):
    return x + 1

f(3)
<!--- end code --->

* If executing this cell produces this error
```
Miyabi kernel: Not logged in to Miyabi.
Open a terminal (File > New > Terminal) and run:
    mount-miyabi
then restart the kernel (Kernel > Restart Kernel).
```
it means the SSH connection between taulec and Miyabi has been lost
* Re-establish it as the message says
  - Open a "Terminal"
  - Run the `mount-miyabi` command
* For this notebook to work, SSH from taulec to Miyabi must succeed without asking for a passphrase or a verification code (more on this later)

## `%%bash` cell

* Python code cells starting with `%%bash` execute the cell content as a shell script, not as Python code
<!--- code w kernel=python --->
%%bash
hostname
<!--- end code --->
* This result confirms that this notebook is running on Miyabi (a login node), not on taulec
* The page you landed on after signing in is served by taulec, but the code in this notebook runs on Miyabi
* More precisely, notebooks using the "Python (Miyabi G)" kernel run on Miyabi (the kernel of a notebook is shown at its top right corner)

* The following command shows your user ID _on Miyabi_
<!--- code w kernel=python --->
%%bash
id
<!--- end code --->
* and this one shows which directory you are in _on Miyabi_
<!--- code w kernel=python --->
%%bash
pwd
<!--- end code --->

## `%%writefile` cell

* Python code cells starting with `%%writefile filename` save the contents of the cell into the specified file when executed
* The content of the cell does not have to be Python code; it can be anything

<!--- code w kernel=python --->
%%writefile hello.c
/* a C cell */
#include <stdio.h>
int main() {
    printf("hello\n");
    return 0;
}
<!--- end code --->

* After saving the above code in `hello.c`, you can compile it by
<!--- code w kernel=python --->
%%bash
gcc -o hello hello.c
<!--- end code --->
and run it by
<!--- code w kernel=python --->
%%bash
./hello
<!--- end code --->

* This is what we typically do in this course
* That is, I give you notebooks containing example code in `%%writefile` cells and command lines in `%%bash` cells

## Text (markdown) cells

* There are also cells for ordinary text (in markdown format), not code

<!--- md w --->
* double-click this cell and edit it
  * when done, press SHIFT+ENTER to render it
<!--- end md --->

# Jupyter Terminals

* Besides notebooks like this page, Jupyter has a plain terminal
* In this course, you will occasionally need it to run commands directly on `taulec`, so get familiar with it
* To open a terminal, click the "+" icon right below the menu to show the Launcher page, and then click "Terminal" (or choose "File" -> "New" -> "Terminal" from the menu)

# Submitting Jobs to Compute Nodes

* This notebook is running on one of the _login nodes_ of the Miyabi supercomputer (`miyabi-g{1,2,..}`)
* There are many ($>$ 1,000) _compute nodes_, on which real jobs are supposed to run
* To run a job on compute node(s), the standard way is to write a shell script (called a _batch job script_) and submit it
* See the Miyabi User's Guide (available from the [Miyabi User Portal](https://miyabi-www.jcahpc.jp/)), Section 5 "Batch Job and Interactive Job", for details
* It is rather long, so here I summarize the minimum you need in this class, and a more convenient method (the `sub` command and the `%%sub` cell magic) I made for this class

## A minimal batch job script

* This is a minimal job script that shows the hostname, the user id, and the current directory of the job
* <font color=red>NOTE:</font> Replace
```
#PBS -q lecture-mig
```
below with
```
#PBS -q lecture1-mig
```
_during lecture time_ (this is a ground rule: `lecture1-xxx` during lecture time and `lecture-xxx` otherwise; more on this later)

<!--- code w kernel=python --->
%%writefile minjob.sh
#!/bin/bash
#PBS -l select=1
#PBS -l walltime=5:00
#PBS -W group_list=gt81
#PBS -q lecture-mig
#PBS -j oe

# --- the main part starts ---
cd "${PBS_O_WORKDIR}"
hostname
id
pwd
<!--- end code --->

* You can submit this script as a job with the `qsub` command

<!--- code w kernel=python --->
%%bash
qsub minjob.sh
<!--- end code --->

* After a while, its standard output and standard error are written to a file named `minjob.sh.oXXXXXXX`

<!--- code w kernel=python --->
%%bash
ls
<!--- end code --->

<!--- code w kernel=python --->
%%bash
cat minjob.sh.o*
<!--- end code --->

* A job may have to wait for a while before it starts, until a compute node becomes available
* You can monitor the status of a job (waiting or running) with `qstat`
* Let's submit a job that takes longer (100 seconds)

<!--- code w kernel=python --->
%%writefile longjob.sh
#!/bin/bash
#PBS -l select=1
#PBS -l walltime=5:00
#PBS -W group_list=gt81
#PBS -q lecture-mig
#PBS -j oe

# --- the main part starts ---
cd "${PBS_O_WORKDIR}"

hostname
id
pwd

for i in $(seq 1 100); do
  date
  sleep 1
done
<!--- end code --->

<!--- code w kernel=python --->
%%bash
qsub longjob.sh
<!--- end code --->

* Execute the following several times to watch the job status

<!--- code w kernel=python --->
%%bash
qstat
<!--- end code --->

* When the job has finished, check its output

<!--- code w kernel=python --->
%%bash
cat longjob.sh.o*
<!--- end code --->

## Queues and other options

### Queues

* I noted that
```
#PBS -q lecture-mig
```
should be
```
#PBS -q lecture1-mig
```
during lecture slots (Monday 16:50-18:35)

* In general, the `-q` option specifies the _queue_ to submit the job to
* Each queue represents compute nodes of a certain type. Specifically,
  - `lecture-mig` : ARM CPU + NVIDIA GPU (GH200), configured with _Multi-Instance GPU (MIG)_
  - `lecture-g` : ARM CPU + NVIDIA GPU (GH200)
  - `lecture-c` : Intel CPU

  outside lecture time, and
  - `lecture1-mig`
  - `lecture1-g`
  - `lecture1-c`

  during lecture time
* In short, MIG splits a single physical GPU into a few smaller GPUs
* We will probably use the `-mig` queues mainly for small experiments and the `-g` queues for large ones
* Even for CPU experiments, we mainly use the ARM CPU, so we will rarely use the `-c` queues
* But we will see as we go

### Other options

* Brief comments on the other options
  * `#PBS -l select=1` says "we are going to use just one node"
  * `#PBS -l walltime=5:00` specifies the time limit (5 minutes)
  * `#PBS -W group_list=gt81` specifies your group; you do not need to (and should never) change it
  * `#PBS -j oe` says "combine standard output and standard error into one file"; without it, you get a separate file for each
* If a job runs longer than the specified `walltime`, it is killed
* Specifying walltime is always a good idea, especially when you know the job is short, because the scheduler may start short jobs earlier (backfilling)
* But for educational use (including this course), the maximum walltime is 15 minutes anyway

## `%%sub` cell

* As you have seen, there is nothing conceptually difficult about `qsub`, but it is a bit tedious and verbose
  * You have to write a job script each time
  * You have to specify many options that rarely change
  * You have to poll until the job finishes
  * Each job leaves an output file, which you have to delete from time to time
* I made a small wrapper around `qsub` and `qstat` for your convenience

* First, load the extension by executing the following
<!--- code w kernel=python --->
%load_ext miyabi
<!--- end code --->

* If you put `%%sub` at the beginning of a cell, it works like `%%bash`, except that it submits the cell as a job with the default options (more on this later), keeps watching the job until it finishes, and shows the job's standard output/error as it goes

<!--- code w kernel=python --->
%%sub 
for i in $(seq 1 10); do
  date
  sleep 1
done
<!--- end code --->

* With `-n` (dry run), you can see the job script that would be submitted, including the options and which file they came from

<!--- code w kernel=python --->
%%sub -n
hostname
<!--- end code --->

* You can override or add just the options you want

<!--- code w kernel=python --->
%%sub
#PBS -q lecture-c
pwd
hostname
<!--- end code --->

* For your convenience, `%%sub --no` does not submit the job, but runs the cell directly on the login node, much like `%%bash`

<!--- code w kernel=python --->
%%sub --no
pwd
hostname
<!--- end code --->

# "Reset Buttons" When Something Goes Wrong in Jupyter

* Jupyter is handy, but prone to errors caused by all sorts of system-level problems, which you can work around but cannot eliminate entirely

## Cell status indicator

* While a cell is executing, the indicator to its left shows `[*]`, which becomes a number like `[3]` when it finishes
* Watch it
<!--- code w kernel=python --->
import time
for i in range(5):
  print(i)
  time.sleep(1)
<!--- end code --->
* While a cell is in this state, no other cell starts executing
* This is important to know, as a cell may fail to finish because of a system-level issue
* When that happens, consider restarting the kernel or the whole server as described below

## Stopping a cell execution

* The `■` button on the top toolbar of a notebook should stop a running cell
* Practice it by executing the above cell again and pressing `■` in the middle
* It often does not work, especially for `%%bash` cells
* Use it as the first, lightweight option when a cell does not finish, without expecting too much from it

## Restarting the kernel

* A stronger method is to restart the kernel
* You can do it from the cycle icon in the top tool bar or the menu "Kernel" -> "Restart Kernel"
* After this, you may have to re-execute some cells (`%load_ext` cells, in particular), as it wipes out the in-memory state of the kernel

## Stopping and starting the server

* This is the biggest reset switch, which restarts the whole thing
* From the menu "File" -> "Hub Control Panel", click "Stop My Server" and then "Start My Server"
* The effect is as if you restarted the kernels of all running notebooks

# Issues Specific to This Course You Don't Want to Know but Have to

* As explained in the [Getting Started with the Course Environment](https://taura.github.io/parallel-distributed-programming/html/get_started.html) page, the setup is a somewhat complex combination of the taulec server and the Miyabi supercomputer
* The idea is that you only access the Jupyter server on taulec from your browser, to fetch/submit assignments and ask AI for help, while your files live on Miyabi, the code in your notebooks runs on Miyabi, and your jobs go to Miyabi's compute nodes
* There are two key pieces that make this work transparently
  * File sharing between taulec and Miyabi, done by [sshfs](https://github.com/libfuse/sshfs)
  * The "Python (Miyabi G)" kernel, a special Jupyter kernel that runs on Miyabi
* Both require that you can `ssh miyabig` from taulec, and both occasionally need your action to keep them working
* I'll cover each of them below
* You don't have to know all the details, but you do have to know the actions required
* This is a diagram showing the overall structure

![overview](svg/overview.svg)

## File sharing between taulec and Miyabi

* From taulec's point of view, the "Fetch" button in Jupyter puts files under taulec's `~/notebooks/pd`
* It is actually a symbolic link to taulec's `~/miyabi/notebooks/pd`, and `~/miyabi` is a mount of Miyabi's `/work/gt81/share/home/t81xxx` (read `t81xxx` as your user name on Miyabi); that is, every access to a file or directory under taulec's `~/miyabi` goes to the corresponding one under Miyabi's `/work/gt81/share/home/t81xxx`
* This mount is done by `sshfs`, roughly like this
```
taulec$ sshfs miyabig:/work/gt81/share/home/t81xxx ~/miyabi
```
but always use the following shortcut instead, which also takes care of logging in to Miyabi
```
taulec$ mount-miyabi
```
* The mount may occasionally break, e.g., when the connection to Miyabi is lost or when the server on taulec is restarted
* Remember the following to fix it when something is broken
  * Open a "Terminal" from the Launcher page of Jupyter
  * `mount-miyabi -s` (or `df ~/miyabi`) tells you whether `~/miyabi` properly mounts the Miyabi directory
  * `mount-miyabi` fixes it if not
* Other things that might be useful to know
  * `mount-miyabi -u` (or `fusermount -u ~/miyabi`) unmounts `~/miyabi`; it should not be necessary, but may help when something does not work even though the mount appears alive according to `df`
  * If it says `~/miyabi` is in use, close the programs it lists (e.g., `cd` out of `~/miyabi` in terminals) and try again

## "Python (Miyabi G)" kernel

* When you click an icon on the Jupyter Launcher page, you normally start a process on the same machine as the server (i.e., taulec)
* The "Python (Miyabi G)" kernel is special in that it runs the process on Miyabi, not on taulec
* As you can imagine, this is done with `ssh` (`ssh miyabig`, to be specific), so you must be able to `ssh miyabig` successfully from `taulec` for "Python (Miyabi G)" to work
* In particular, `ssh miyabig` must succeed _without any interactive input (without being asked for a passphrase or a verification code)_
* This is made possible by the following settings (`ControlXXX`), which I have put in your `~/.ssh/config` on `taulec`
```
Host miyabig
    HostName miyabi-g.jcahpc.jp
    User t81xxx
    ControlMaster auto
    ControlPath ~/.ssh/cm_%r@%h:%p
    ControlPersist 6h
    ServerAliveInterval 60
```
* Check it by opening a "Terminal" and running `cat ~/.ssh/config`
* With these settings, once you log in with `mount-miyabi` (or `ssh miyabig`) from `taulec`, the authenticated connection is kept, and you are not asked for the passphrase or the verification code again while it lasts; it lasts while it is in use (e.g., while `~/miyabi` is mounted) and for six hours (6h) after that
* Most importantly, if executing a cell fails with this error:
```
Miyabi kernel: Not logged in to Miyabi.
Open a terminal (File > New > Terminal) and run:
    mount-miyabi
then restart the kernel (Kernel > Restart Kernel).
```
run `mount-miyabi` in a terminal as it says; it logs you in to Miyabi again (asking for the passphrase and the verification code) and re-mounts `~/miyabi` if needed
* Notebooks you "Fetch" are set to use the "Python (Miyabi G)" kernel, so they run on Miyabi when opened

# (Optional) SSH from your PC to Miyabi

* You may prefer logging in to Miyabi from your PC and working on files directly, without going through Jupyter (e.g., when you edit files with VS Code)
* It is perfectly OK to do so if you find it more comfortable
* Here are things to remember:
  * Register the public key corresponding to the private key on your PC at the [Miyabi User Portal](https://miyabi-www.jcahpc.jp/), in addition to the one you made on taulec
  * You still have to fetch, read, and submit exercises from Jupyter (taulec)
  * You may also want to run `%%writefile` cells in Jupyter, to save example code
  * You cannot run a coding agent on Miyabi login nodes, so make sure to run `opencode` on `taulec`, somewhere under `~/notebooks/pd/`
  * Fetched notebooks and the example code you save go under `/work/gt81/share/home/t81xxx/notebooks/pd` on Miyabi
  * Keep your work under it on Miyabi, so that the files are visible from `taulec` and, more importantly, you can submit them later via Jupyter

# (Optional) SSH from your PC to taulec

* You may also want to SSH to taulec from your PC, though there are fewer reasons to do so
* Remember that Jupyter has "Terminal", so direct SSH is unnecessary for occasional command-line operations
* Also remember that the "real" files are on Miyabi and taulec simply mounts them, so if you want to edit files with your favorite editor rather than Jupyter, you probably want to SSH to Miyabi, not taulec
* But you may still prefer to SSH to taulec, or need to do so for troubleshooting, which is of course possible
* To do so, add a line with the public key corresponding to your PC's private key, such as this
```
ssh-ed25519 AAAAC3.........................................CSii4 you@local_pc
```
to `taulec`'s `~/.ssh/authorized_keys`
* A possible procedure:
  * click the upload icon of the file browser and upload the public key (e.g., `id_xxxxx.pub`) to your home directory (the top folder)
  * run the following in a "Terminal" (_not in this notebook_, which runs on Miyabi, not taulec!); it works whether or not `~/.ssh/authorized_keys` exists
```
# check if it exists and its contents (OK if it does not exist)
taulec$ cat ~/.ssh/authorized_keys
 ...
# add (append) the public key
taulec$ cat id_xxxxx.pub >> ~/.ssh/authorized_keys
```

