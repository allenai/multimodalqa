## **A Question Understanding Benchmark**

Break is a question understanding dataset, aimed at training models to reason over complex questions. It
features [83,978](https://github.com/allenai/Break) natural language questions, annotated with a new meaning representation, Question
Decomposition Meaning Representation (QDMR). Each example has the natural question along with its QDMR representation. Break contains human
composed questions, sampled from [10 leading question-answering benchmarks](#question-answering-datasets) over text, images and databases.
This dataset was created by a team of [NLP researchers](#authors) at [Tel Aviv University](https://www.tau-nlp.org/)
and [Allen Institute for AI](https://allenai.org/).

For more details on Break, please refer to our [TACL 2020 paper](#paper), and to
our [blogpost](https://allenai.github.io/Break/blogpost.html).

<center>
    <a href="https://allenai.github.io/Break/images/qdmr01.png"> 
        <img src="images/qdmr01.png" height="170">
      </a>
</center>

## **Question-Answering Datasets**

The Break dataset contains questions from the following 10 datasets:

* **Semantic Parsing**: [Academic](https://github.com/jkkummerfeld/text2sql-data), [ATIS](https://github.com/jkkummerfeld/text2sql-data)
  , [GeoQuery](https://github.com/jkkummerfeld/text2sql-data), [Spider](https://yale-lily.github.io/spider)
* **Visual Question Answering**: [CLEVR-humans](https://cs.stanford.edu/people/jcjohns/clevr/), [NLVR2](http://lil.nlp.cornell.edu/nlvr/)
* **Reading Comprehension (and KB-QA)**: [ComQA](http://qa.mpi-inf.mpg.de/comqa/), [ComplexWebQuestions](https://www.tau-nlp.org/compwebq)
  , [DROP](https://allennlp.org/drop), [HotpotQA](https://hotpotqa.github.io/)

For the full dataset statistics please refer to our [repository](https://github.com/allenai/Break).

## **Paper**

[**Break It Down: A Question Understanding Benchmark**](https://arxiv.org/abs/2001.11770v1)  
Tomer Wolfson, Mor Geva, Ankit Gupta, Matt Gardner, Yoav Goldberg, Daniel Deutch and Jonathan Berant  
*Transactions of the Association for Computational Linguistics (TACL), 2020*

```markdown
@article{Wolfson2020Break,
  title={Break It Down: A Question Understanding Benchmark},
  author={Wolfson, Tomer and Geva, Mor and Gupta, Ankit and Gardner, Matt and Goldberg, Yoav and Deutch, Daniel and Berant, Jonathan},
  journal={Transactions of the Association for Computational Linguistics},
  year={2020},
}
```

## **Authors**

> Talent wins games, but teamwork and intelligence wins championships.

*Michael Jordan*

<div>
<div class="card">
  <img src="images/authors/author_01.jpg" alt="Avatar" style="width:100%">
  <div class="container">
    <a href="https://tomerwolgithub.github.io/">
    <h4><b>Tomer Wolfson</b></h4>  
    </a>
  </div>
</div>
<div class="card">
  <img src="images/authors/author_02.jpg" alt="Avatar" style="width:100%">
  <div class="container">
    <a href="https://mega002.github.io/">
    <h4><b>Mor <br>Geva</b></h4>  
    </a>
  </div>
</div>
<div class="card">
  <img src="images/authors/author_03.jpg" alt="Avatar" style="width:100%">
  <div class="container">
    <a href="https://sites.google.com/view/ag1988/home">
    <h4><b>Ankit Gupta</b></h4>  
    </a>
  </div>
</div>
<div class="card">
  <img src="images/authors/author_04.jpg" alt="Avatar" style="width:100%">
  <div class="container">
    <a href="https://allenai.org/team/mattg/">
    <h4><b>Matt Gardner</b></h4>  
    </a>
  </div>
</div>
<div class="card">
  <img src="images/authors/author_05.jpg" alt="Avatar" style="width:100%">
  <div class="container">
    <a href="https://www.cs.bgu.ac.il/~yoavg/uni/">
    <h4><b>Yoav Goldberg</b></h4>  
    </a>
  </div>
</div>
<div class="card">
  <img src="images/authors/author_06.jpg" alt="Avatar" style="width:100%">
  <div class="container">
    <a href="https://www.cs.tau.ac.il/~danielde/">
    <h4><b>Daniel Deutch</b></h4>  
    </a>
  </div>
</div>
<div class="card">
  <img src="images/authors/author_07.jpg" alt="Avatar" style="width:100%">
  <div class="container">
    <a href="http://www.cs.tau.ac.il/~joberant/">
    <h4><b>Jonathan Berant</b></h4>  
    </a>
  </div>
</div>
</div>

## **Leaderboard**

### **Submission**

Evaluating predictions for the hidden test set is done via the [AI2 Leaderboard page](https://leaderboard.allenai.org/). Log on to the
leaderboard website and follow the submission instructions.

* **[Break Leaderboard](https://leaderboard.allenai.org/break/)**
* **[Break High-Level Leaderboard](https://leaderboard.allenai.org/break_high_level/)**

*Given the GED metric is computed by an approximation algorithm, the evaluation may take several hours. The approximation algorithm also
results in slightly different GED values than the paper.*

### **Results**

**Break**

Rank | Submission | Created | EM Dev. | EM Test | SARI Dev. | SARI Test | GED Dev. | GED Test
------------ | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | -------------
1 | Curriculum-trained CopyNet <br>*Chris Coleman and Alex Reneau,*<br>*Northwestern
University* | Jul 1, 2020 | **`_`**  | **`0.163`** | **`_`**  | **`0.757`** | **`_`**  | **`0.271`**
2 | CopyNet <br>*([Wolfson et al., TACL 2020](https://arxiv.org/abs/2001.11770v1))* | Feb 1, 2020 | **`0.154`**  | `0.157` | **`0.748`**  | `0.746` | **`0.318`**  | `0.322`
3 | RuleBased <br>*([Wolfson et al., TACL 2020](https://arxiv.org/abs/2001.11770v1))* | Feb 1, 2020 | `0.002`  | `0.003` | `0.508`  | `0.506` | `0.799`  | `0.802`

**Break High-level**

Rank | Submission | Created | EM Dev. | EM Test | SARI Dev. | SARI Test | GED Dev. | GED Test
------------ | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | -------------
1 | CopyNet <br>*([Wolfson et al., TACL 2020](https://arxiv.org/abs/2001.11770v1))* | Feb 1, 2020 | **`0.081`**  | **`0.083`** | **`0.722`**  | **`0.722`** | **`0.319`**  | **`0.316`**
2 | RuleBased <br>*([Wolfson et al., TACL 2020](hhttps://arxiv.org/abs/2001.11770v1))* | Feb 1, 2020 | `0.010`  | `0.012` | `0.554`  | `0.554` | `0.659`  | `0.652`

## **Explore**

To view (many) more question decomposition examples, [explore Break](/explore.md).

## **Download**

- For the full documentation of the dataset and its format please refer to our [Github repository](https://github.com/allenai/Break).
- Click here to [download Break](https://github.com/allenai/Break/raw/master/break_dataset/Break-dataset.zip).
