from utils import *

parser = argparse.ArgumentParser(description='Choose your model(s) & language(s)')
parser.add_argument('--model',type=str,
                    help='Provide the model you want to use. Check and choose from the key values of the MODEL_PATHS variable. If you want to test on multiple models, provide multiple model names with ", " between each (e.g., "gpt-4-0125-preview, aya-101").')
parser.add_argument('--language',type=str,default=None,
                    help='Provide the language you want to test on. Check and choose from the first values of the LANG_COUNTRY variable. If you want to test on multiple languages, provide multiple languages with ", " between each (e.g., "English, Korean").')
parser.add_argument('--country',type=str,default=None,
                    help='Provide the country you want to test on. Check and choose from the second values of the LANG_COUNTRY variable. If you want to test on multiple countries, provide multiple countries with ", " between each (e.g., "UK, South Korea"). Make sure you have the same number of countries and languages provided. The language-country pair do not have to be identical with the pairs within the LANG_COUNTRY variable.')
parser.add_argument('--question_dir',type=str,default=None,
                    help='Provide the directory name with (translated) questions.')
parser.add_argument('--question_file',type=str,default=None,
                    help='Provide the csv file name with (translated) questions.')
parser.add_argument('--question_col',type=str,default=None,
                    help='Provide the column name from the given csv file name with (translated) questions.')
parser.add_argument('--prompt_dir',type=str,default=None,
                    help='Provide the directory where the propmts are saved.')
parser.add_argument('--prompt_file',type=str,default=None,
                    help='Provide the name of the csv file where the propmts are saved.')
parser.add_argument('--prompt_no',type=str,default=None,
                    help='Provide the propmt id (ex. inst-1, inst-2, pers-1, etc.)')
parser.add_argument('--id_col',type=str,default="ID",
                    help='Provide the column name from the given csv file name with question IDs.')
parser.add_argument('--output_dir',type=str,default='./model_inference_results',
                    help='Provide the directory for the output files to be saved.')
parser.add_argument('--output_file',type=str,default=None,
                    help='Provide the name of the output file.')
parser.add_argument('--model_cache_dir',type=str,default='.cache',
                    help='Provide the directory saving model caches.')
parser.add_argument("--gpt_azure", type=str2bool, nargs='?',
                    const=True, default=False,
                    help="Whether you are using the AzureOpenAI for GPT-models' response generation.")
parser.add_argument('--temperature',type=int,default=0,
                    help='Provide generation temperature for GPT models.')
parser.add_argument('--top_p',type=int,default=0,
                    help='Provide generation top_p for GPT models.')

args = parser.parse_args()

def make_prompt(question,prompt_no,language,country,prompt_sheet):
    prompt = prompt_sheet[prompt_sheet['id']==prompt_no]
    if language == 'English':
        prompt = prompt['English'].values[0]
    else:
        prompt = prompt['Translation'].values[0]

    return prompt.replace('{q}',question)

def generate_response(model_name,model_path,tokenizer,model,language,country,q_df,q_col,id_col,output_dir,prompt_no=None,prompt_sheet=None):
    replace_country_flag = False
    if language != COUNTRY_LANG[country] and language == 'English':
        replace_country_flag = True
        
    if q_col == None:
        if language == COUNTRY_LANG[country]:
            q_col = 'Translation'
        elif language == 'English':
            q_col = 'Question'
    
    if prompt_no is not None:
        if prompt_sheet is None:
            # Load prompts from local CSV file
            if args.prompt_dir and args.prompt_file:
                prompt_file_path = os.path.join(args.prompt_dir, args.prompt_file)
            else:
                prompt_file_path = os.path.join('../data/prompts', f'{country}_prompts.csv')
            if os.path.exists(prompt_file_path):
                prompt_sheet = pd.read_csv(prompt_file_path, encoding='utf-8')
            else:
                raise FileNotFoundError(f"Prompt file not found: {prompt_file_path}")
        output_filename = os.path.join(output_dir,f"{model_name}-{country}_{language}_{prompt_no}_result.csv")
    else:
        output_filename = os.path.join(output_dir,f"{model_name}-{country}_{language}_result.csv")
    print(q_df[[id_col,q_col]])
    
    guid_list = set()
    if os.path.exists(output_filename):
        already = pd.read_csv(output_filename)
        guid_list = set(already[id_col])
        print(already)
        
        
    else:        
        write_csv_row([id_col,q_col,'prompt','response','prompt_no'],output_filename)
      
    pb = tqdm(q_df.iterrows(),desc=model_name,total=len(q_df))
    for _,d in pb:
        q = d[q_col]
        guid = d[id_col]
        pb.set_postfix({'ID':guid})
        
        if guid in guid_list:
            continue
       
        if replace_country_flag:
            q = replace_country_name(q,country.replace('_',' '))
       
        if prompt_no is not None:
            prompt = make_prompt(q,prompt_no,language,country,prompt_sheet)
        else:
            prompt = q
            
        print(prompt)
        
        response = get_model_response(model_path,prompt,model,tokenizer,temperature=args.temperature,top_p=args.top_p,gpt_azure=args.gpt_azure)
            
        print(response)
        write_csv_row([guid,q,prompt,response,prompt_no],output_filename)
        
    del guid_list
            
def get_response_from_all():
    models = args.model
    languages = args.language
    countries = args.country
    question_dir = args.question_dir
    question_file = args.question_file
    question_col = args.question_col
    prompt_no = args.prompt_no
    id_col = args.id_col
    output_dir = args.output_dir
    azure = args.gpt_azure
    
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    
    
    if ',' in languages:
        languages = languages.split(',')
        
    if ',' in countries:
        countries = countries.split(',')
        
    if ', ' in models:
        models = models.split(',')
        
    if type(languages) == type(countries) and isinstance(languages,list):
        if len(languages) != len(countries):
            print("ERROR: Same number of languages and countries necessary. If multiple languages and countries are given, each element of the two lists should be in pairs.")
            exit()
        
    def get_questions(language,country):
        # Try different possible filenames
        possible_files = [
            os.path.join(question_dir, f'{country}_full_final_questions.csv'),
            os.path.join(question_dir, f'{country}_questions.csv'),
            os.path.join(question_dir, question_file) if question_file else None
        ]
        for file_path in possible_files:
            if file_path and os.path.exists(file_path):
                questions_df = pd.read_csv(file_path, encoding='utf-8')
                return questions_df
        raise FileNotFoundError(f"Question file not found for country {country}. Tried: {possible_files}")
    
    
    def generate_response_per_model(model_name):
        model_path = MODEL_PATHS[model_name]
        
        tokenizer,model = get_tokenizer_model(model_name,model_path,args.model_cache_dir)
        
        # Load prompt sheet once if needed
        prompt_sheet = None
        if prompt_no is not None:
            if isinstance(countries, str):
                country_for_prompt = countries
            else:
                country_for_prompt = countries[0] if countries else None
            if country_for_prompt:
                if args.prompt_dir and args.prompt_file:
                    prompt_file_path = os.path.join(args.prompt_dir, args.prompt_file)
                else:
                    prompt_file_path = os.path.join(question_dir.replace('questions', 'prompts'), f'{country_for_prompt}_prompts.csv')
                if os.path.exists(prompt_file_path):
                    prompt_sheet = pd.read_csv(prompt_file_path, encoding='utf-8')
        
        if isinstance(languages,str):
            questions = get_questions(languages,countries)
            generate_response(model_name,model_path,tokenizer,model,languages,countries,questions,question_col,id_col,output_dir,prompt_no=prompt_no,prompt_sheet=prompt_sheet)
        else:
            for l,c in zip(languages,countries):
                questions = get_questions(l,c)
                # Load prompt sheet for this country
                if prompt_no is not None and prompt_sheet is None:
                    prompt_file_path = os.path.join(question_dir.replace('questions', 'prompts'), f'{c}_prompts.csv')
                    if os.path.exists(prompt_file_path):
                        prompt_sheet = pd.read_csv(prompt_file_path, encoding='utf-8')
                generate_response(model_name,model_path,tokenizer,model,l,c,questions,question_col,id_col,output_dir,prompt_no=prompt_no,prompt_sheet=prompt_sheet)
        
    if isinstance(models,str):
       generate_response_per_model(models)
    else:
        for m in models:
            generate_response_per_model(m)

 
if __name__ == "__main__":
    get_response_from_all()    