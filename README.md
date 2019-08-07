# My cool new model

## Abstract

This is what the model uses & what it does.

## Getting Started

Create a conda environment with python interpreter of choice.
```bash
conda create --name pytorch1.1 python=3.7.3
conda activate pytorch1.1
```
Install all dependencies using `conda install`.

```bash
cd name-of-model
mkdir datasets/
```
Add your datasets into `datasets/`.

### Training

Start the `visdom` server.  
```bash
visdom
```

Run the project.  

```bash
python main.py --phase train
```

### Testing

```bash
python main.py --phase test --in_path xxx/xxx.jpg  --out_path ./results/
```

## Results

This is what my cool new model achieved in testing.
Here are some graphs and tables.

## References

* [Some awesome paper](www.github.com/rshwndsz)
* [Great blog](www.github.com/rshwndsz)
