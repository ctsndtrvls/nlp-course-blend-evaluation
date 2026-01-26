from evaluation_utils import *

import unicodedata as ud
from string import punctuation

# Make language-specific imports optional
# pip install konlpy
try:
    from konlpy.tag import Okt
    KONLPY_AVAILABLE = True
except ImportError:
    KONLPY_AVAILABLE = False
    Okt = None

# pip install hausastemmer
try:
    import hausastemmer
    HAUSASTEMMER_AVAILABLE = True
except ImportError:
    HAUSASTEMMER_AVAILABLE = False
    hausastemmer = None

# git clone https://github.com/aznlp-disc/stemmer.git, cp word.txt & suffix.txt.
try:
    from stemmer.stemmer import Stemmer as AZStemmer
    AZSTEMMER_AVAILABLE = True
except ImportError:
    AZSTEMMER_AVAILABLE = False
    AZStemmer = None

# pip install nlp-id
try:
    from nlp_id.lemmatizer import Lemmatizer as IDLemmatizer
<<<<<<< Updated upstream
    IDLEMMATIZER_AVAILABLE = True
except ImportError:
    IDLEMMATIZER_AVAILABLE = False
    IDLemmatizer = None
=======
except ImportError:
    IDLemmatizer = None

>>>>>>> Stashed changes

# pip install hazm
try:
    from hazm import Lemmatizer as PRLemmatizer
    PRLEMMATIZER_AVAILABLE = True
except ImportError:
    PRLEMMATIZER_AVAILABLE = False
    PRLemmatizer = None

# pip install qalsadi
try:
    from qalsadi.lemmatizer import Lemmatizer as ARLeammatizer
    ARLEMMATIZER_AVAILABLE = True
except ImportError:
    ARLEMMATIZER_AVAILABLE = False
    ARLeammatizer = None

# pip install cltk
try:
    from cltk import NLP
    CLTK_AVAILABLE = True
except ImportError:
    CLTK_AVAILABLE = False
    NLP = None

# !pip install spark-nlp==5.3.3 pyspark==3.3.1
try:
    from sparknlp.base import *
    from sparknlp.annotator import *
    from sparknlp.pretrained import PretrainedPipeline
    import sparknlp
    SPARKNLP_AVAILABLE = True
except ImportError:
    SPARKNLP_AVAILABLE = False

# SUSTEM for Sundanese
try:
    from SUSTEM.SUSTEM_S import *
    SUSTEM_AVAILABLE = True
except ImportError:
    SUSTEM_AVAILABLE = False

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    spacy = None

# pip install jieba
try:
    import jieba
    JIEBA_AVAILABLE = True
except ImportError:
    JIEBA_AVAILABLE = False
    jieba = None

# git clone https://github.com/anoopkunchukuttan/indic_nlp_library.git & https://github.com/anoopkunchukuttan/indic_nlp_resources.git
# The path to the local git repo for Indic NLP library
INDIC_NLP_LIB_HOME=os.path.abspath("./indic_nlp_library")

# The path to the local git repo for Indic NLP Resources
INDIC_NLP_RESOURCES=os.path.abspath("./indic_nlp_resources")

INDICNLP_AVAILABLE = False
try:
    sys.path.append(INDIC_NLP_LIB_HOME)
    from indicnlp import common
    from indicnlp import loader
    from indicnlp.tokenize import indic_tokenize
    INDICNLP_AVAILABLE = True
except (ImportError, OSError):
    INDICNLP_AVAILABLE = False  



def lemma_check(answer,llm_response,nlp_pipeline,language='Korean'):
    if answer in llm_response or answer.replace('-',' ') in llm_response or answer.replace(' ','-') in llm_response:
        return True
    
    if language == 'Korean':
        if not KONLPY_AVAILABLE:
            return False  # Cannot process Korean without konlpy
        okt = Okt()
        answer_tokens = okt.morphs(' '.join([w for w,p in okt.pos(answer) if p!='Josa']),stem=True)
        llm_tokens = okt.morphs(' '.join([w for w,p in okt.pos(llm_response) if p!='Josa']),stem=True)
        
    elif language == 'Hausa':
        if not HAUSASTEMMER_AVAILABLE:
            return False  # Cannot process Hausa without hausastemmer
        answer_tokens = [hausastemmer.stem(term.strip('-')) for term in answer.split()]
        llm_tokens = [hausastemmer.stem(term.strip('-')) for term in llm_response.split()]
    
    elif language == 'Amharic':
        answer_tokens = [token.result if lemma.result.startswith('_') else lemma.result for token,lemma in zip(nlp_pipeline.fullAnnotate(answer)[0]['lemma'],nlp_pipeline.fullAnnotate(answer)[0]['token'])]
        llm_tokens = [token.result if lemma.result.startswith('_') else lemma.result for token,lemma in zip(nlp_pipeline.fullAnnotate(llm_response)[0]['lemma'],nlp_pipeline.fullAnnotate(llm_response)[0]['token'])]
        
    elif language == 'Azerbaijani':
        if not AZSTEMMER_AVAILABLE:
            return False  # Cannot process Azerbaijani without AZStemmer
        # Instantiate Stemmer object
        my_stemmer = AZStemmer()
        
        def stem_words(my_text):
            my_text=my_text.replace("İ", "I")
            my_text=my_text.replace("“", "")
            my_text=my_text.replace("”", "")
            my_text=my_text.replace("'", "")
            my_text=my_text.replace('"', "")
            my_text=my_text.split()
            my_words=[]
            for word in my_text:
                my_words.append(''.join(c for c in word if (c not in punctuation) or (c == '-')))
            # Apply stemming to the list of words
            my_words = my_stemmer.stem_words(my_words)
            # Print words after stemming
            return my_words
        
        answer_tokens = stem_words(answer)
        llm_tokens = stem_words(llm_response)
    
    elif language == 'Indonesian':
        if not IDLEMMATIZER_AVAILABLE:
            return False  # Cannot process Indonesian without IDLemmatizer
        lemmatizer = IDLemmatizer() 
        answer_tokens = lemmatizer.lemmatize(answer).split()
        llm_tokens = lemmatizer.lemmatize(llm_response).split() 
    
    elif language == 'Persian':
        if not PRLEMMATIZER_AVAILABLE:
            return False  # Cannot process Persian without PRLemmatizer
        lemmatizer = PRLemmatizer()
        answer_tokens = [lemmatizer.lemmatize(term) for term in answer.split()]
        llm_tokens = [lemmatizer.lemmatize(term) for term in llm_response.split()]
        
    elif language == 'Arabic':
        if not ARLEMMATIZER_AVAILABLE:
            return False  # Cannot process Arabic without ARLeammatizer
        lemmatizer = ARLeammatizer()
        answer_tokens = lemmatizer.lemmatize(answer)
        llm_tokens = lemmatizer.lemmatize(llm_response) 
        
    elif language == 'Greek':
        cltk_nlp = NLP(language="grc", suppress_banner=True)
        answer_tokens = cltk_nlp.analyze(text=answer).lemmata
        llm_tokens = cltk_nlp.analyze(text=llm_response).lemmata
        
    elif language == 'Spanish':
        if nlp_pipeline is not None and SPARKNLP_AVAILABLE:
            answer_tokens = [lemma.result for lemma in nlp_pipeline.fullAnnotate(answer)[0]['lemma']]
            llm_tokens = [lemma.result for lemma in nlp_pipeline.fullAnnotate(llm_response)[0]['lemma']]
        else:
            # Fall back to simple word matching if SparkNLP not available
            answer_tokens = answer.lower().split()
            llm_tokens = llm_response.lower().split()
        
    elif language == 'Sundanese':
        stemmer = EcsStemmer()
        answer_tokens = [stemmer.stemmingProcess(word.replace('(','').replace(')','')) for word in answer.split()]
        llm_tokens = [stemmer.stemmingProcess(word.replace('(','').replace(')','')) for word in llm_response.split()]

        
    elif language == 'English':
        if not SPACY_AVAILABLE or nlp_pipeline is None:
            # Fall back to simple word matching if spacy not available
            answer_tokens = answer.lower().split()
            llm_tokens = llm_response.lower().split()
        else:
            answer_tokens = [token.lemma_ for token in nlp_pipeline(answer)]
            llm_tokens = [token.lemma_ for token in nlp_pipeline(llm_response)]
        
    elif language == 'Chinese':
        if not JIEBA_AVAILABLE:
            return False  # Cannot process Chinese without jieba
        answer_tokens = list(jieba.cut(answer))
        llm_tokens = list(jieba.cut(llm_response))
        
    elif language == 'Assamese':
        common.set_resources_path(INDIC_NLP_RESOURCES)
        loader.load()
        
        answer_tokens = indic_tokenize.trivial_tokenize(answer)
        llm_tokens = indic_tokenize.trivial_tokenize(llm_response)
        
    d = {ord('\N{COMBINING ACUTE ACCENT}'):None}
    
    answer_tokens = [ud.normalize('NFD',term).translate(d).lower() for term in answer_tokens if term not in punctuation and term != '']
    llm_tokens = [ud.normalize('NFD',term).translate(d).lower() for term in llm_tokens if term not in punctuation and term != '']
    
    for a in answer_tokens:
        if a not in llm_tokens:
            return False        
    
    return True

def hard_exact_match(annotation_dict,response_df,id_col,r_col,annotations_key='annotations'):
    binary_score = 0
    weight_score = 0
    
    for qid,data in annotation_dict.items():
        llm_response = get_llm_response_by_id(response_df,qid,id_col,r_col)
        
        if llm_response and data[annotations_key]:
            max_vote = max(list(data[annotations_key].values()))
        
            for k,v in sorted(data[annotations_key].items(), key=lambda item: item[1],reverse=True):
                if k == llm_response:
                    binary_score += 1
                    weight_score += v/max_vote
                    break
            
    binary_score = binary_score / len(annotation_dict) * 100
    weight_score = weight_score / len(annotation_dict) * 100
    
    print(binary_score)
    print(weight_score)
    
    return binary_score, weight_score

def soft_exact_match(country,language,annotation_dict,response_df,id_col,r_col,annotations_key='aggregated_answers'):
    binary_score = 0
    weight_score = 0
    valid_question_cnt = 0
    
    if language == 'Spanish':
        if SPARKNLP_AVAILABLE:
            try:
                spark = sparknlp.start()
                
                document_assembler = DocumentAssembler() \
                    .setInputCol("text") \
                    .setOutputCol("document")

                tokenizer = Tokenizer() \
                    .setInputCols(["document"]) \
                    .setOutputCol("token")

                lemmatizer = LemmatizerModel.pretrained("lemma", "es") \
                        .setInputCols(["token"]) \
                        .setOutputCol("lemma")
                        
                nlp_pipeline = Pipeline(stages=[document_assembler, tokenizer, lemmatizer])
                nlpPipeline = LightPipeline(nlp_pipeline.fit(spark.createDataFrame([['']]).toDF('text')))
            except Exception as e:
                print(f"Warning: Could not initialize SparkNLP for Spanish: {e}")
                print("Falling back to simple word matching for Spanish")
                nlpPipeline = None
        else:
            print("Warning: SparkNLP not available. Using simple word matching for Spanish")
            nlpPipeline = None
    
    elif language == 'Amharic':
        spark = sparknlp.start()
        
        document_assembler = DocumentAssembler() \
            .setInputCol("text") \
            .setOutputCol("document")

        tokenizer = Tokenizer() \
            .setInputCols(["document"]) \
            .setOutputCol("token")

        lemmatizer = LemmatizerModel.pretrained("lemma", "am") \
                .setInputCols(["token"]) \
                .setOutputCol("lemma")

        nlp_pipeline = Pipeline(stages=[document_assembler,tokenizer,lemmatizer])
        nlpPipeline = LightPipeline(nlp_pipeline.fit(spark.createDataFrame([['']]).toDF('text')))
    
    else:
        nlpPipeline = None
        
    # Load English lemmatizer if available
    if SPACY_AVAILABLE:
        try:
            en_lemmatizer = spacy.load("en_core_web_sm")
        except (OSError, IOError):
            # Model not installed, use None
            en_lemmatizer = None
            print("Warning: en_core_web_sm model not found. English evaluation will use simple word matching.")
    else:
        en_lemmatizer = None
        print("Warning: spacy not available. English evaluation will use simple word matching.")
        
    response_df['binary_score'] = [None]*response_df.shape[0]
    response_df['weight_score'] = [None]*response_df.shape[0]
    
    pb = tqdm(annotation_dict.items(),total=len(annotation_dict))
    
    for qid,data in pb:
        pb.set_description(qid)
        if data['idks']['no-answer']+data['idks']['not-applicable'] >= 3 or data['idks']['idk']>=5 or len(data[annotations_key])==0:
            continue
        
        valid_question_cnt += 1
        
        llm_response = get_llm_response_by_id(response_df,qid,id_col,r_col)
        flag = False
        if llm_response and data[annotations_key]:
            max_vote = data[annotations_key][0]['count']
            
            for agg_ans in data[annotations_key]:
                if language != 'English':
                    for a in agg_ans['answers']:
                        if lemma_check(a,llm_response,nlpPipeline,language):
                            binary_score += 1
                            weight_score += agg_ans['count']/max_vote
                            flag = True
                            break
                if not flag:
                    for a in agg_ans['en_answers']:
                        if lemma_check(a,llm_response,en_lemmatizer,'English'):
                            binary_score += 1
                            weight_score += agg_ans['count']/max_vote
                            flag = True
                            break
                if flag:
                    break
        if flag:
            response_df.loc[response_df[id_col]==qid,'binary_score'] = 1
            response_df.loc[response_df[id_col]==qid,'weight_score'] = agg_ans['count']/max_vote
            print(response_df.loc[response_df[id_col]==qid])
        else:
            response_df.loc[response_df[id_col]==qid,'binary_score'] = 0
            response_df.loc[response_df[id_col]==qid,'weight_score'] = 0
            
        pb.set_postfix({'bs':binary_score/valid_question_cnt*100,'ws':weight_score/valid_question_cnt*100})
            
    binary_score = binary_score / valid_question_cnt * 100
    weight_score = weight_score / valid_question_cnt * 100
    
    print(binary_score)
    print(weight_score)
    
    return binary_score, weight_score, response_df