from tqdm import tqdm
from utils import saveData, loadData, preprocess
from utils import BASELINE_ROOT, MODEL_ROOT, BASELINE_DATA, MODEL_DATA

INPUT_LENGTH = 30
TARGET_LENGTH = 10
ANS_LENGTH = 5


############################
# Preprocess baseline data #
############################
def remove_cols(df):
    """
    Drop unecessary columns
    """
    df.drop('Asin', inplace=True, axis=1)
    df.drop('AnswerTime', inplace=True, axis=1)
    df.drop('UnixTime', inplace=True, axis=1)
    df.drop('Category', inplace=True, axis=1)
    df.drop('AnswerType', inplace=True, axis=1)

    df.dropna()
    df.reset_index(drop=True, inplace=True)
    return df


def add_len_col(df):
    """
    Process Dataframe
    """
    df.dropna()
    df['Question'] = preprocess(df['Question'])
    df['Answer'] = preprocess(df['Answer'])
    df.drop_duplicates(inplace=True)
    df.dropna()
    df.reset_index(drop=True, inplace=True)

    # add QA length
    df['QuestionLength'] = df['Question'].str.split().apply(len)
    df['AnswerLength'] = df['Answer'].str.split().apply(len)

    # remove QA with length 0
    df = df[df['AnswerLength'] > 0]
    df = df[df['QuestionLength'] > 0]
    df.reset_index(drop=True, inplace=True)
    return df


def restrict_len(df, ans_length):
    """
    Resctrict the Questions and Answers length
    - Questions length <= mean length
    - Answers length <= Anslength
    """
    ques_length = int(df['QuestionLength'].mean())
    df = df[df["QuestionLength"] <= ques_length]
    df.reset_index(drop=True, inplace=True)
    df = df[df['AnswerLength'] <= ans_length]
    df.reset_index(drop=True, inplace=True)
    # add begin & end indicators to answers, add QA column
    df['Answer'] = df['Answer'].apply(lambda x: '<bos> ' + str(x) + ' <eos>')
    df['QA'] = df['Question'].astype(str) + ' ' + df['Answer'].astype(str)
    return df


#########################
# Preprocess model data #
#########################
def vectorization(df):
    """
    Vectorize the data and split out the input and target texts.
    """
    input_texts, target_texts = [], []
    input_words_set, target_words_set = set(), set()
    for conversation_index in tqdm(range(df.shape[0])):
        if conversation_index == 0:
            continue
        input_text = df.iloc[conversation_index - 1]
        target_text = df.iloc[conversation_index]
        if input_text.conversation_id == target_text.conversation_id:
            input_text = input_text.message
            target_text = target_text.message
            if input_text and target_text \
                    and len(input_text.split()) in range(3, INPUT_LENGTH) \
                    and len(target_text.split()) in range(1, TARGET_LENGTH):
                # Add <bos> and <eos> indicators to the target_text
                target_text = '<bos> ' + target_text + ' <eos>'
                input_texts.append(input_text)
                target_texts.append(target_text)
                for word in input_text.split():
                    if word not in input_words_set:
                        input_words_set.add(word)
                for word in target_text.split():
                    if word not in target_words_set:
                        target_words_set.add(word)
    return input_texts, target_texts, input_words_set, target_words_set


def save_vec_result(df, root):
    """
    Save the vectorization result.
    """
    input_texts, target_texts, input_words_set, target_words_set = vectorization(df)
    fname1, fname2, fname3, fname4 = '/input_texts.pickle', '/target_texts.pickle', \
                                     '/input_words_set.pickle', '/target_words_set.pickle'
    saveData(input_texts, root + fname1)
    saveData(target_texts, root + fname2)
    saveData(input_words_set, root + fname3)
    saveData(target_words_set, root + fname4)
    return input_texts, target_texts, input_words_set, target_words_set


def process_df(df, dataset_type):
    """
    Further process the datasets
    """
    if dataset_type == 'BASELINE':
        df = remove_cols(df)
        df = add_len_col(df)
        df = restrict_len(df, ANS_LENGTH)
    if dataset_type == 'MODEL':
        df.message = preprocess(df.message)
        df.dropna()
        df.drop('sentiment', inplace=True, axis=1)
        df.reset_index(drop=True, inplace=True)
    return df


if __name__ == '__main__':
    baseline_data = loadData(BASELINE_DATA)
    model_data = loadData(MODEL_DATA)

    baseline_df = process_df(baseline_data, 'BASELINE')
    model_df = process_df(model_data, 'MODEL')

    fname1 = '/processed_clean_single_qna.csv'
    saveData(baseline_df, BASELINE_ROOT + fname1)
    fname2 = '/cleaned_topical_chat.csv'
    saveData(model_df, MODEL_ROOT + fname2)

    input_texts, target_texts, input_words_set, target_words_set = save_vec_result(model_df, './MyModelDataFiles')
    print(len(input_texts), len(target_texts), len(input_words_set), len(target_words_set))
