#* Parallel and Distributed Programming Hands-on Environment

<!--- md w --->

Enter your name and student ID.

 * Name:
 * Student ID:

<!--- end md --->

# How this page works (Jupyter notebook basics)

* This page is a Jupyter notebook
* It consits of texts and code cells

## Cell

* A textbox like below is called a "cell"
* Press SHIFT+ENTER to execute it

## Python code cell

<!--- code w kernel=python --->
def f(x):
    return x + 1

f(3)
<!--- end code --->

* If executing this cell generates an error saying
```
Miyabi kernel: Not logged in to Miyabi.
Open a terminal (File > New > Terminal) and run:
    mount-miyabi
then restart the kernel (Kernel > Restart Kernel).
```
it means SSH connection between taulec and Miyabi is dead
* Re-establish it by doing the above
  - Launch "Terminal"
  - Execute `mount-miyabi` command
* In order for this notebook to work, SSH from taulec to Miyabi has to succeed without password being asked

## `%%bash` cell

* Python code cells starting with `%%bash` executes the cell content as a shell script, not Python code
<!--- code w kernel=python --->
%%bash
hostname
<!--- end code --->
* From this result, you can confirm that this notebook is executing on Miyabi (login node), not on taulec
* The page you landed after sign in is running on taulec, but this notebook is running on Miyabi
* To be more precise, notebooks run by "Python (Miyabi G)" kernel is executing on Miyabi (check the top right corner of the notebook to know which kernel is running the notebook)

* The following command shows your user ID _on Miyabi_
<!--- code w kernel=python --->
%%bash
id
<!--- end code --->
* and which directory you are in _on Miyabi_
<!--- code w kernel=python --->
%%bash
pwd
<!--- end code --->

## `%%writefile` cell

* Python code cells starting with `%%writefile filename`, when executed, saves the contents of the cell into the specified file
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

* After saving the above code in `hello.c`, you can then compile it by
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

* there are cells for ordinary texts (markdown format), not code

<!--- md w --->
* double-click this cell and edit
  * after done, press SHIFT-ENTER to save
<!--- end md --->

# Jupyter Terminals

* Besides notebooks like this page, Jupyter has a plain terminal
* In particular, in this course, you have to use it occasionally to run commands on `taulec` directly, so become familiar with it
* To launch a terminal, click the "+" icon right below the menu to show the launcher page
* Click "Terminal"

# Submitting Jobs to Compute Nodes

* The notebook is running on one of the _login nodes_ of Miyabi supercomputer (`miyabi-g{1,2,..}`)
* There are many ($>$ 1,000) _compute nodes_, on which real jobs are supposed to run
* To send a job to a compute node(s), the canonical way is to write a shell script (called a _batch job script_)
* You can see Miyabi User's Guide (available from [Miyabi User Portal](https://miyabi-www.jcahpc.jp/)) Section 5 Batch Job and Interactive Job for details
* But it is rather long, so I will summarize the minimum basics necessary in this class, and a more convenient method (`sub` command and `%%sub` cell magic) I have made for this class

## The minimum batch job script 

* This is the minimum job script that shows hostname, user id, and the current directory of the job
* <font color=red>NOTE:</font> Replace 
```
#PBS -q lecture-mig
```
below to
```
#PBS -q lecture1-mig
```
_during lecture time_ (this is a ground rule.  `lecture1-xxx` in lecture time and `lecture-xxx` not during lecture time. more on this later)

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

* You can submit a job executing this script by `qsub` command

<!--- code w kernel=python --->
%%bash
qsub minjob.sh
<!--- end code --->

* After a while, its standard output and standard error are written files named `minjob.sh.oXXXXXXX`

<!--- code w kernel=python --->
%%bash
ls
<!--- end code --->

<!--- code w kernel=python --->
%%bash
cat minjob.sh.o*
<!--- end code --->

* A job may take a while until it starts execution to wait for a compute node to become available
* You can monitor the job status (waiting or running) by `qstat`
* Let's submit a job that takes long (100 seconds)

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

* Execute the following multiple times to watch the job status

<!--- code w kernel=python --->
%%bash
qstat
<!--- end code --->

* When the job is finished, check the output

<!--- code w kernel=python --->
%%bash
cat longjob.sh.o*
<!--- end code --->

## Queue types and other options

### Queue types

* I have noted that 
```
#PBS -q lecture-mig
```
should be 
```
#PBS -q lecture1-mig
```
during lecture slots (Monday 16:50-18:35) 

* In general `-q` option specifies _queue name_ to submit the job to
* Each queue represents compute nodes of a certain type. Specifically,
  - `lecture-mig` : ARM CPU + NVIDIA GPU (GH200), configured with _Multi-Instance GPU (MIG)_
  - `lecture-g` : ARM CPU + NVIDIA GPU (GH200)
  - `lecture-c` : Intel CPU
not during lecture time and
  - `lecture-mig`
  - `lecture-g`
  - `lecture-c`
during lecture time
* In short, MIG is a mini GPU inside a single physical GPU
* We will probably use `-mig` queue mainly for small experiments and `-g` for large experiments
* Even for CPU experiments, we mainly use ARM CPU, so we won't be using `-c` queues
* But we will see along the way

### Other options

* Brief comments about other options
  * `#PBS -l select=1` says "we are going to use just one node"
  * `#PBS -l walltime=5:00` specifies time limit (5 minutes)
  * `#PBS -W group_list=gt81` specifies the group you are in; you do not have to and should never change it
  * `#PBS -j oe` says "combine standard output and standard error into one file"; without it, you'll get a separate file for each
* If the job execution time exceeds the specified `walltime`, then the job is killed
* Specifying walltime is always a good idea, especially when you know it's short, because the scheduler may prefer short jobs (backfilling)
* But for educational use (including this course), the maximum walltime is 15 minutes anyway

## `%%sub` cell

* As you have seen, there is nothing conceptually demanding about `qsub`, but it is obviously a bit tedious, verbose, and cumbersome
  * You have to write a job script each time
  * You have to specify lot of rarely changing options
  * You have to poll when the job gets finished
  * Each job spits a file and you have to delete them manually from time to time
* I have made a small wrapper around `qsub` and `qstat` for your convenience

* First load the extension by executing the following
<!--- code w kernel=python --->
%load_ext miyabi
<!--- end code --->

* If you put `%%sub` in the beginning of a cell, it is like `%%bash`, except it submits the job with default options (more on this later), keeps watching the job status until the job gets finished, showing job standard output/error along the way

<!--- code w kernel=python --->
%%sub 
for i in $(seq 1 10); do
  date
done
<!--- end code --->

* You can see options used and which file the options came from with `-n` (dry run) option

<!--- code w kernel=python --->
%%sub -n
hostname
<!--- end code --->

* You may overwrite/add only options you want to overwrite/add

<!--- code w kernel=python --->
%%sub -n
#PBS -q lecture-c
pwd
hostname
<!--- end code --->

* For your convenience, `%%sub --no` does not send the job to the queue but executes the cell directly on the login node, much like `%%bash`

<!--- code w kernel=python --->
%%sub --no
pwd
hostname
<!--- end code --->

# "Reset Buttons" When Something Went Wrong in Jupyter

* Jupyter is nice but prone to errors associated with all sorts of system-level errors you can only avoid, not eliminate entirely

## Cell status indicator

* While a cell is executing, it is marked `[*]` on the status indicator left of the cell, which becomes a number like `[3]` when finished
* Watch it
<!--- code w kernel=python --->
import time
for i in range(5):
  print(i)
  time.sleep(1)
<!--- end code --->
* While a cell is in this state, you cannot start executing another cell
* This is important to know, as a cell might fail to finish due to a system level issue
* When that happens, consider restarting the kernel or the whole session as described below

## Stopping a cell execution

* `■` button should stop a cell during execution, but it often does not work
* Use it as the first lightweight option when a cell does not finish, without expecting too much from it

## Restart the kernel 

* A stronger method is "Reset the kernel" 
* You can invoke it from the menu "Kernel" -> "Restart Kernel"
* After this, you might have to re-execute cells (`%load_ext` cells, in particular) as it wipes out in-memory state of the kernel

## Stop and start server

* This is the biggest reset switch to restart the whole thing
* You can invoke it from the menu "File" -> "Hub Control Panel"
* The effect is as if you reset the kernel of every running pages

# Issues Specific to This Course You Don't Want to Know but You Have to

* As explained in this [Getting Started with the Course Environment](https://taura.github.io/parallel-distributed-programming/html/get_started.html) page, the exact setting is a bit complex combination of taulec server and Miyabi supercomputer
* The idea is you just access taulec Jupyter server from your browser to fetch/submit assignments and ask AI for help, while you are making files on Miyabi, running Jupyter pages on Miyabi, and sending jobs to compute nodes
* There are two key elements to accomplish this transparency or integration
  * File sharing between taulec and Miyabi, accomplished by [sshfs](https://github.com/libfuse/sshfs)
  * "Python (Miyabi G)" kernel, a special Jupyter kernel to execute on Miyabi
* Both require you to be able to `ssh miyabig` from taulec and both require your action occasionally to maintain the proper state
* I'll cover each of them below
* You don't have to know all the details, but you have to know actions required

## File sharing between taulec and Miyabi

* From taulec's perspective, "Fetch" button in Jupyter puts files under taulec's `~/notebooks/pd`
* Which is actually a symbolic link to taulec's `~/miyabi/notebooks/pd`, and `~/miyabi` mounts Miyabi's `/work/gt81/share/home/t81xxx` (where `t81xxx` should be read as your user name on Miyabi), which means all file/directory accesses under taulec's `~/miyabi` go to the corresponding file/directory under Miyabi's `/work/gt81/share/home/t81xxx`
* This "`~/miyabi` mounts Miyabi's `/work/gt81/share/home/t81xxx`" is accomplished by `sshfs`; in short, you should have done
```
taulec$ sshfs miyabig:/work/gt81/share/home/t81xxx ~/miyabi
```
or its convenient shortcut:
```
taulec$ mount-miyabi
```
* This mount state may occasionally be broken, e.g., when you restart the Jupyter server
* Remember the following
  * launch "Terminal" from the launcher page of Jupyter
  * `df` or `df ~/miyabi` command tells you whether `~/miyabi` properly mounts the Miyabi directory
  * `mount-miyabi` fixes it if not
* Other things that might be useful to know
  * `fusermount -u ~/miyabi` or its convenient shortcut `mount-miyabi -u` forcefully unmounts `~/miyabi`; should not be necessary but may be required when something does not work properly even when the mount state appears to be alive according to `df`

## "Python (Miyabi G)" kernel

* When you click an icon from the Jupyter launcher page, you normally run a process in the same node as the server (i.e., taulec)
* "Python (Miyabi G)" kernel is special in that it runs the process on Miyabi, not taulec
* As you can imagine, it is done by `ssh` (more specifically, `ssh miyabig`), so you must be able to `ssh miyabig` successfuly from `taulec` in order for "Python (Miyabi G)" to work
* In particular, it requires `ssh miyabig` to succeed _without interactive user input (without passphrase or verfication code being asked)_
* The way to make that possible is the following set of configurations (`ControlXXX`) , which I have put for you in your `~/.ssh/config` file on `taulec`
```
Host miyabig
     HostName miyabi-g3.jcahpc.jp
     User t81xxx
     ControlMaster auto
     ControlPath ~/.ssh/cm_%r@%h:%p
     ControlPersist 6h
```
* Confirm it by launching "Terminal" and execute `cat ~/.ssh/config`
* With this setting, once you successfully `ssh miyabig` from `taulec`, the authenticated channel will be kept for another six hours (6h), during which you do not have to input verification code or passphrase
* More important, if a cell fails to execute a cell with this error:
```
Miyabi kernel: Not logged in to Miyabi.
Open a terminal (File > New > Terminal) and run:
    mount-miyabi
then restart the kernel (Kernel > Restart Kernel).
```
you should first execute `ssh miyabig` and input passphrase and/or verification code, or as suggested, `mount-miyabi` does it as a part of establishing the mount operation
* When you "Fetch" exercise materials, these notebooks are configured to run with "Python (Miyabi G)" kernel when opened, so it automatically runs on Miyabi

# (Optional) SSH from your PC to Miyabi

* You may prefer to logging in Miyabi from your local PC and working on files directly without going through Jupyter (e.g., when you edit files with VSCode)
* It's perfectly OK to do so if you feel it's more comfortable
* Here are things to remember:
  * Register the public key corresponding the private key in your PC, in [Miyabi User Portal](https://miyabi-www.jcahpc.jp/)
  * You still have to fetch, read, and submit exercises from Jupyter (taulec)
  * You may also want to run `%%writefile` cells in Jupyter, to save example code
  * You can't run coding agent on Miyabi login node, so make sure run `opencode` on `taulec`, somewhere under `~/notebooks/pd/`
  * Fetched notebooks and saved example code go under `/work/gt81/share/home/t81xxx/notebooks/pd` on Miyabi
  * Keep your work under it on Miyabi, so those files are visible from `taulec`, and more importantly, you can submit them later via Jupyter

# (Optional) SSH from your PC to taulec

* You may also want to SSH taulec from local PC, thought there are less reasons to do so
* Remember that you have "Terminal" in Jupyter, so direct SSH is unnecessary for occasional command line operations
* Also remember that "real" files are hosted on Miyabi and which taulec is simply mounting, so if you want to work on files with your favorite editor not Jupyter, you presumably want to SSH Miyabi, not taulec
* But you may still prefer to SSH taulec or need to do so for troubleshooting, which is of course possible
* To do so, just add a line of the public key corresponding to your local PC's private key, such as this
```
ssh-ed25519 AAAAC3.........................................CSii4 you@local_pc
```
in `taulec`'s `~/.ssh/authorized_keys`
* A possible exact procedure:
  * click the upload icon and upload the public key (e.g., `id_xxxxx.pub`)
  * execute the following from "Terminal" (_not on this page_, which is executing on Miyabi, not taulec!), which works both when `~/.ssh/authorized_keys` does or does not exist
```
# check if it exsits and its contents (OK if it does not exist)
taulec$ cat ~/.ssh/authorized_keys
 ...
# add (append) the public key
taulec$ cat id_xxxxx.pub >> ~/.ssh/authorized_keys
```

