
# Multi-Modal QA Dataset Format

The dataset is constructed from QA files and context files:
1) `MultiModalQA_train/dev/test.jsonl.gz` - contains questions and answers, for train, dev and test set respectively
2) `tables.jsonl.gz` - contains the tables contexts
3) `texts.jsonl.gz` - contains the texts contexts
4) `images.jsonl.gz` - contains the metadata of the images contexts
5) `images` - a directory contains the images contexts

# QA Files Format

Each line of the files `MultiModalQA_train/dev.jsonl.gz` contains one question, 
along with its answers, metadata and supporting context pointers (context supports the answers for this questions)

```json
{
  "qid": "5454c14ad01e722c2619b66778daa98b",
  "question": "who owns the rights to little shop of horrors?",
  "answers": ["answer1", "answer2"],
  "metadata": {},
  "supporting_context": [{
      "doc_id": "46ae2a8e7928ed5a8e5f9c59323e5e49",
      "doc_part": "table"
    },
    {
      "doc_id": "d57e56eff064047af5a6ef074a570956",
      "doc_part": "image"
    }]
}
```

`MultiModalQA_test.jsonl.gz` contains lines with the same format as above, only that the `answers` 
and `supporting_context` fields are removed.

## A Single Answer

Each answer in `answers` field of each question contains a string or a yesno answer,
along with specification of the supporting context of that answer:

```json
{
  "answer": "AnswerText",
  "type": "string/yesno",
  "modality": "text/image/table",
  "text_instances": [{
          "doc_id": "b95b35eabfc80a0f1a8fd8455cd6d109",
          "part": "text",
          "start_byte": 345,
          "text": "AnswerText"
        }],
  "table_indices": [[5, 2]],
  "image_instances": [{
              "doc_id": "d57e56eff064047af5a6ef074a570956",
              "doc_part": "image"
            }]
}
```

## A Single Question Metadata

The metadata of each question contains its type, modalities required to solve it, the wiki entities appear 
in the question and in the answers, the machine generated question (the question before human rephrasing), 
metadata about the rephrasing process, a list of texts docs and image docs that are part of the context for
this question (supporting context + distractors), a pointer to the related table, 
and a list of intermediate answers (the answers of the sub-questions composing the multi-modal question)  

```json
{
    "type": "Compose(TableQ,ImageListQ)",
    "modalities": [
      "image",
      "table"
    ],
    "wiki_entities_in_question": [
      {
        "text": "Domenico Dolce",
        "wiki_title": "Domenico Dolce",
        "url": "https://en.wikipedia.org/wiki/Domenico_Dolce"
      }
    ],
    "wiki_entities_in_answers": [],
    "pseudo_language_question": "In [Members] of [LGBT billionaires] what was the [Net worth USDbn](s) when the [Name] {is completely bald and wears thick glasses?}",
    "rephrasing_meta": {
      "accuracy": 1.0,
      "edit_distance": 0.502092050209205,
      "confidence": 0.7807520791930855
    },
    "image_doc_ids": [
      "89c1b7c3c061cc80bb98d99cbbec50dd",
      "0f3858e2186b2030b77c759fc727e20b"
    ],
    "text_doc_ids": [
      "498369348c988d866b5fac0add45bac5",
      "57686242cf542e30cbad13037017b478"
    ],
    "intermediate_answers": ["answer1", "answer2"],  # provided in the same format described above
    "table_id": "46ae2a8e7928ed5a8e5f9c59323e5e49"
  }
```

# A Single Table Format

Each line of `tables.jsonl.gz` represents a single table. `table_rows` is a list of rows, where each row
is a list of cells. Each cell is provided with its text and wiki entities. `header` provides for each column in the talble
its name along with parsing metadata extracted from it such as NERs and items type. 

```json
{
  "title": "Dutch Ruppersberger",
  "url": "https://en.wikipedia.org/wiki/Dutch_Ruppersberger",
  "id": "dcd7cb8f23737c6f38519c3770a6606f",
  "table": {
    "table_rows": [
      [
        {
          "text": "Baltimore County Executive",
          "links": [
            {
              "text": "Baltimore County Executive",
              "wiki_title": "Baltimore County Executive",
              "url": "https://en.wikipedia.org/wiki/Baltimore_County_Executive"
            }
          ]
        },
      ]
    ],
    "table_name": "Electoral history",
    "header": [
      {
        "column_name": "Year",
        "metadata": {
          "parsed_values": [
            1994.0,
            1998.0
          ],
          "type": "float",
          "num_of_links": 9,
          "ner_appearances_map": {
            "DATE": 10,
            "CARDINAL": 1
          },
          "is_key_column": true,
          "entities_column": true
        }
      }
    ]
  }
}
```

# A Single Image Metadata Format

Each line in `images.jsonl.gz` holds metadata for a single image. The `path` provided points to the image file
in the provided images directory.

```json
{
  "title": "Taipei",
  "url": "https://en.wikipedia.org/wiki/Taipei",
  "id": "632ea110be92836441adfb3167edf8ff",
  "path": "Taipei.jpg"
}
```

# A Single Text Metadata Format
Each line in `texts.jsonl.gz` represents a single text paragraph. 

```json
{
  "title": "The Legend of Korra (video game)",
  "url": "https://en.wikipedia.org/wiki/The_Legend_of_Korra_(video_game)",
  "id": "16c61fe756817f0b35df9717fae1000e",
  "text": "Over three years after its release, the game was removed from sale on all digital storefronts on December 21, 2017."
}
```
