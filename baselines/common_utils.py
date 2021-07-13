
ALL_QUESTION_TYPES = [
    'TextQ',
    'TableQ',
    'ImageQ',
    'ImageListQ',
    'Compose(TableQ,ImageListQ)',
    'Compose(TextQ,ImageListQ)',
    'Compose(ImageQ,TableQ)',
    'Compose(ImageQ,TextQ)',
    'Compose(TextQ,TableQ)',
    'Compose(TableQ,TextQ)',
    'Intersect(TableQ,TextQ)',
    'Intersect(ImageListQ,TableQ)',
    'Intersect(ImageListQ,TextQ)',
    'Compare(Compose(TableQ,ImageQ),TableQ)',
    'Compare(Compose(TableQ,ImageQ),Compose(TableQ,TextQ))',
    'Compare(TableQ,Compose(TableQ,TextQ))',
]

TEXT_SINGLE_HOP_QUESTION_TYPES = [
    'TextQ',
]
TEXT_AS_FIRST_HOP_QUESTION_TYPES = [
    'Compare(TableQ,Compose(TableQ,TextQ))',
    'Compose(ImageQ,TextQ)',
    'Compose(TableQ,TextQ)',
    'Intersect(TableQ,TextQ)',
    'Intersect(ImageListQ,TextQ)',
]
TEXT_AS_SECOND_HOP_QUESTION_TYPES = [
    'Compare(Compose(TableQ,ImageQ),Compose(TableQ,TextQ))',
    'Compose(TextQ,ImageListQ)',
    'Compose(TextQ,TableQ)',
]

TABLE_SINGLE_HOP_QUESTION_TYPES = [
    "TableQ"
]
TABLE_AS_FIRST_HOP_QUESTION_TYPES = [
    'Compose(ImageQ,TableQ)',
    'Compose(TextQ,TableQ)',
]
TABLE_AS_SECOND_HOP_QUESTION_TYPES = [
    'Compare(Compose(TableQ,ImageQ),TableQ)',
    'Compare(TableQ,Compose(TableQ,TextQ))',
    'Compose(TableQ,ImageListQ)',
    'Compose(TableQ,TextQ)',
    'Intersect(ImageListQ,TableQ)',
    'Intersect(TableQ,TextQ)',
]

IMAGE_SINGLE_HOP_QUESTION_TYPES = [
    'ImageQ',
    'ImageListQ'
]
IMAGE_AS_FIRST_HOP_QUESTION_TYPES = [
    'Compare(Compose(TableQ,ImageQ),Compose(TableQ,TextQ))',
    'Compare(Compose(TableQ,ImageQ),TableQ)',
    'Compose(TableQ,ImageListQ)',
    'Compose(TextQ,ImageListQ)',
    'Intersect(ImageListQ,TableQ)',
]
IMAGE_AS_SECOND_HOP_QUESTION_TYPES = [
    'Compose(ImageQ,TableQ)',
    'Compose(ImageQ,TextQ)',
    'Intersect(ImageListQ,TextQ)',
]


# every question should be answered either as a single hop question, or two-hop question
assert set(TEXT_SINGLE_HOP_QUESTION_TYPES + TEXT_AS_SECOND_HOP_QUESTION_TYPES
           + TABLE_SINGLE_HOP_QUESTION_TYPES + TABLE_AS_SECOND_HOP_QUESTION_TYPES
           + IMAGE_SINGLE_HOP_QUESTION_TYPES + IMAGE_AS_SECOND_HOP_QUESTION_TYPES) == set(ALL_QUESTION_TYPES)
assert len(set(TEXT_SINGLE_HOP_QUESTION_TYPES) & set(TEXT_AS_SECOND_HOP_QUESTION_TYPES)) == 0
assert len(set(TABLE_SINGLE_HOP_QUESTION_TYPES) & set(TABLE_AS_SECOND_HOP_QUESTION_TYPES)) == 0
assert len(set(IMAGE_SINGLE_HOP_QUESTION_TYPES) & set(IMAGE_AS_SECOND_HOP_QUESTION_TYPES)) == 0

MULTI_HOP_QUESTION_TYPES = TEXT_AS_SECOND_HOP_QUESTION_TYPES \
                           + TABLE_AS_SECOND_HOP_QUESTION_TYPES + \
                           IMAGE_AS_SECOND_HOP_QUESTION_TYPES
# no duplicated multi-hop question types
assert len(MULTI_HOP_QUESTION_TYPES) == len(set(MULTI_HOP_QUESTION_TYPES))
# no duplication for the first hop
assert set(TEXT_AS_FIRST_HOP_QUESTION_TYPES + TABLE_AS_FIRST_HOP_QUESTION_TYPES + IMAGE_AS_FIRST_HOP_QUESTION_TYPES) \
       == set(MULTI_HOP_QUESTION_TYPES)


def process_question_for_implicit_decomp(question, question_type, hop=0, bridge_entity='', sep_token='[SEP]'):
    if isinstance(bridge_entity, list) or isinstance(bridge_entity, set):
        bridge_entity = "; ".join(bridge_entity)
    return (
        f'{question_type} {sep_token} '
        f'HOP={hop} {sep_token} '
        f'{bridge_entity} {sep_token} '
        f'{question}')


def extract_numbers_from_str(s):
    numbers = []
    for token in s.split():
        try:
            num = int(token.replace(",", ""))
        except:
            try:
                num = float(token)
            except:
                num = None
        if num:
            numbers.append(num)
    return numbers