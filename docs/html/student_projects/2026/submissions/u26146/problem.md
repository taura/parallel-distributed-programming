# A tiny Transformer for router command generation

## Problem

A router has settings such as a Wi-Fi channel, a DNS server, and a guest network.
This program learns to turn a short, fixed-format English request into a sequence
of configuration commands. For example:

```text
Request: set guest wifi on, set channel 11, set dns 1.1.1.1,
         firewall strict, tx power 70, max clients 16

Target:  set wifi.guest=1; set wifi.channel=11; set wan.dns=1.1.1.1;
         set firewall.mode=strict; set wifi.tx_power=70;
         set wifi.max_clients=16; commit;
```

These commands are a synthetic teaching format which looks similar to the real FRR/vtysh commands but cannot operate on the real routers.
The goal is to study the computation and performance of a small language model.

## Data and task size

The program creates its own training and test data. It does not use an external dataset.

Each example contains a request and its expected command. The program chooses one value for each setting:

| Setting | Possible values | Command rule |
|---|---|---|
| Guest Wi-Fi | on, off | Convert on to `1` and off to `0` |
| Wi-Fi channel | 1, 6, 11 | Copy the selected number |
| DNS server | 1.1.1.1, 8.8.8.8, 9.9.9.9, 8.8.4.4 | Copy the selected address |
| Firewall mode | normal, strict | Copy the selected mode |
| Transmit power | 40, 70, 100 | Copy the selected number |
| Maximum clients | 8, 16, 32, 64 | Copy the selected number |

The commands always appear in the same order and end with `commit;`. These are made-up commands for this experiment, not verified commands for a real router.

## Example

**Request:**

```text
set guest wifi on, set channel 6, set dns 1.1.1.1, firewall strict, tx power 70, max clients 16
```

**Expected command:**

```text
set wifi.guest=1; set wifi.channel=6; set wan.dns=1.1.1.1; set firewall.mode=strict; set wifi.tx_power=70; set wifi.max_clients=16; commit;
```

The complete example is stored in this format:

```text
[INTENT] request [CMD] expected command
```

The code creates the expected command using the rules above. The model then tries to learn this conversion by predicting one command character at a time.

## Training and test split

There are **576 possible combinations**:

```text
2 × 3 × 4 × 2 × 3 × 4 = 576
```

The program separates these into **460 training combinations** and **116 test combinations**. The same combination cannot belong to both groups.

Samples are randomly drawn from each group, and repetitions are allowed. Therefore, `train_samples=12000` means 12,000 samples drawn from the training group, not 12,000 different configurations.

Each example is stored as `[INTENT] request [CMD] command` followed by a newline.
A character is one token. The default context window contains 256 characters, enough for the longest
complete example in this task. A shorter window can remove a requested setting
before the model has generated it.

## Model and algorithm

The C++17 program uses only the standard library. It has one attention head,
one block, and a small feed-forward network (FFN). Model width is 48, FFN width
is 96, and vocabulary size is 45 including padding. There are 35,229 parameters
at context 256, or 24,477 at context 32.

The main calculation steps are:

1. Turn characters and their positions into learned number vectors.
2. Calculate keys and values at each position and a query at the last position.
3. Compare the query with keys. Softmax turns scores into attention weights;
   use these weights to combine the values. Padding is ignored.
4. Add the result to the last input vector, apply the FFN with ReLU, and add
   another residual connection. Then predict the next character's probabilities.

Only earlier characters are inputs. This model calculates only the last query's
attention, not a full square attention matrix, and has no LayerNorm.

Training supplies the correct earlier characters (teacher forcing). Loss is
calculated on the command and final newline. Backpropagation finds gradients,
gradient clipping limits large changes, and serial SGD updates the weights.
Invalid numerical values in gradients or output normalization cause an error.
One step selects one sample and updates weights for each command character;
it is not an epoch.

## Serial baseline result (Miyabi run)

The following result is from a serial run on Miyabi. The settings were:

| Setting | Value |
|---|---:|
| Training samples | 12,000 |
| Test samples | 116 |
| Training steps | 1,000 |
| Context length | 256 |
| Model width / FFN width | 48 / 96 |
| Learning rate | 0.005 |
| Gradient clipping limit (`clip_norm`) | 5.0 |
| Random seed | 42 |
| Maximum samples for exact command evaluation | 116 |

All reported measurements from this run are listed below.

| Measurement | Result |
|---|---:|
| Training time (`training_time_ms`) | 166,173.283 ms (166.173283 s) |
| Character-level weight updates | 140,431 |
| Updates with gradient clipping | 77,629 |
| Training character updates per second | 845.088 |
| Test command teacher-forced loss | 1.286 |
| Test command next-character accuracy | 0.506 (50.6%) |
| Exact command match rate | 0.000 (0/116) |
| Teacher-forced evaluation time | 7,247.528 ms |
| Generation time | 11,332.144 ms |

Training time includes context construction, forward/backward calculations,
clipping, updates, and logging; it excludes compilation, data creation, testing,
and generation. Three printed request/answer examples allowed manual checking.
Generated commands repeated `set set set ...`.

### What the result means

Training loss fell from about 2.81 for the first 100 steps to 1.38 for the last
100 steps. Some character patterns appear to be learned, but 50.6% accuracy
uses the correct previous characters. During generation, the model uses its own
outputs, so errors may vary from time to time. 

I do not yet know whether model structure, training settings, or an implementation
issue is responsible. Testing whether it can learn one example would help.
The model generated some character patterns when given the correct preceding characters, but it failed to generate complete commands on its own.

## Build and run

```sh
c++ -O3 -std=c++17 app.cc -o app
./app --train-samples 12000 --test-samples 116 --steps 1000 \
  --ctx 256 --d-model 48 --d-ff 96 --lr 0.005 --clip-norm 5 \
  --seed 42 --print-every 100 --eval-exact-n 116
```

### Command-line options

| Option | Description |
|---|---|
| `--train-samples` | Number of training samples |
| `--test-samples` | Number of test samples |
| `--steps` | Number of training sample selections |
| `--ctx` | Context length |
| `--d-model` | Model dimension |
| `--d-ff` | Feed-forward hidden dimension |
| `--lr` | Learning rate |
| `--clip-norm` | Maximum global gradient norm |
| `--seed` | Random seed |
| `--print-every` | Logging interval in training steps |
| `--eval-exact-n` | Maximum number of samples for exact-match evaluation |

Increasing `--steps` gives more training. Increasing model widths or context length changes the model and usually increases computation. For a speed comparison, keep these settings fixed.

## Opportunities for later parallelization

Parallelism opportunities (the baseline is fully serial)

* Key/value projections across context positions
* Independent output rows of projections
* Attention score calculations
* Dot products and softmax need reductions

Training samples are not independent: each character changes weights used by
the next character. Mini-batches could provide more parallel work but would
change the update method. Generated characters also depend on earlier outputs.

Key/value calculations grow with context length and the square of model width.
The program also recalculates overlapping contexts. These may be costly, but
profiling is needed. SIMD and better memory access may help; tiny parallel tasks
may add overhead instead.

Parallelization will not automatically fix command generation though.
