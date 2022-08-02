# ReDEE

Source code for the paper ["RAAT: Relation-Augmented Attention Transformer for Relation Modeling in Document-Level Event Extraction"](link) , in NAACL 2022.

### Overview

In document-level event extraction (DEE) task, event arguments always scatter across sen- tences (across-sentence issue) and multiple events may lie in one document (multi-event issue). In this paper, we argue that the rela- tion information of event arguments is of great significance for addressing the above two is- sues, and propose a new DEE framework which can model the relation dependencies, called Relation-augmented Document-level Event Ex- traction (ReDEE). More specifically, this frame- work features a novel and tailored transformer, named as Relation-augmented Attention Trans- former (RAAT). RAAT is scalable to cap- ture multi-scale and multi-amount argument relations. To further leverage relation in- formation, we introduce a separate event re- lation prediction task and adopt multi-task learning method to explicitly enhance event extraction performance. Extensive experi- ments demonstrate the effectiveness of the proposed method, which can achieve state-of- the-art performance on two public datasets.

* Architecture

![architecture](/pictures/architecture.png)

* Overall Result

![architecture](/pictures/overall_result.png)

### Code Structure

```
RAAT/
├─ dee/
    ├── __init__.py
    ├── event_type.py: event definition and structure.
    ├── base_task.py: 
    ├── dee_task.py
    ├── ner_task.py
    ├── dee_helper.py: data features construction and evaluation utils
    ├── dee_metric.py: data evaluation utils
    ├── dee_model.py: ReDEE model
    ├── ner_model.py
    ├── transformer.py: transformer module
    ├── utils.py: utils
├─ dre/
    ├── __init__.py
    ├── utils.py
    ├── modeling_bert.py
├─ scripts/
    ├── train_multi.sh
    ├── eval.sh
├─ Data/
    ├── data.zip
├─ README.md
├─ run_dee_task.py: the main entry
├─ requirements.txt
├─ Exps/: experiment outputs
├─ LICENSE
```

### Environment

```
python (3.6.9)
cuda (10.1)
Ubuntu-18.0.4 or centos 7.0
```

### Dependencies

```
torch==1.6.0
pytorch-pretrained-bert==0.4.0
transformers==3.0.2
numpy
tensorboardX
```

### Data Preparation

```
$ cd Data
$ unzip data.zip // you can get train.json, dev.json, and test.json files used in experiment.
```

### Train

```
$ cd scripts
$ ./train_multi.sh
```

### Evaluation

```
$ cd scripts
$ ./eval.sh
```

### Licence

This project is licensed under the Apache License - see the [LICENSE](LICENSE) file for details.

### Citation

If you use this work or code, please kindly cite the following paper:

```
@inproceedings{liang-etal-2022-raat,
    title = "{RAAT}: Relation-Augmented Attention Transformer for Relation Modeling in Document-Level Event Extraction",
    author = "Liang, Yuan  and
      Jiang, Zhuoxuan  and
      Yin, Di  and
      Ren, Bo",
    booktitle = "Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies",
    month = jul,
    year = "2022",
    address = "Seattle, United States",
    publisher = "Association for Computational Linguistics",
    pages = "4985--4997",
}
```

