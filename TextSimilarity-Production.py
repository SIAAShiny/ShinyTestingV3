
# coding: utf-8

# In[1]:

def cleanText(data1):
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    from nltk.stem import WordNetLemmatizer
    import re
    import pandas as pd
    import numpy as np
    stop_words = set(stopwords.words('english'))
    stopwords_to_replace = pd.read_excel('stopwordsToReplace.xlsx')
    stopWordsPruned = [x for x in stop_words if x not in stopwords_to_replace['Col1'].tolist() ]
    short_description_new = []
    
    for i in range(data1.shape[0]):
        xx = data1['Text'][i]
        for j in range(stopwords_to_replace.shape[0]):
            xx = re.sub(r"(\b("+stopwords_to_replace['Col1'][j]+r")\b)",stopwords_to_replace['Col2'][j],str(xx),re.IGNORECASE)
        short_description_new.append(xx)   
    data1['Text'] = short_description_new
    

    wnl = WordNetLemmatizer()
    short_description_new = []

    testing = data1['Text'].astype(str)
    testing = testing.fillna('')
    testing = testing.str.lower()
    testing = testing.apply(lambda x: re.sub('[^a-zA-Z0-9]',' ',x).strip())
    testing = testing.apply(lambda x: re.sub('[0-9]','',x))
    testing = testing.str.replace('.',' ')

#db[x2] = db[x2].str.lower()
    cleanedText1 = []
    cleanedText2 = []

    for i in range(len(testing)):
        word_tokens = word_tokenize(testing[i])
        filtered_text = [w for w in word_tokens if not w in stopWordsPruned]
        filtered_sentence = []

        for w in word_tokens:
            if (w not in stopWordsPruned):
                filtered_sentence.append(w)
        cleanedText1.append(" ".join([k for k in filtered_sentence]))
        cleanedText2.append(" ".join([wnl.lemmatize(j) for j in cleanedText1[i].split()]))


    return cleanedText2


# In[2]:

def Similarity_Criteria(data1,criteria,MAX_CAT,Max_Iter):
    import pandas as pd
    import numpy as np
    
    text_to_be_used_later = data1['Text'].tolist()
    data1['Text'] = cleanText(data1)
    data1['Category'] = data1['Category'].astype(str)
    y = data1['Category'].tolist()
    cols_in_data_final = [w.replace('[','_') for w in y]
    cols_in_data_final = [w.replace(']','_') for w in cols_in_data_final]
    cols_in_data_final = [w.replace('<','_') for w in cols_in_data_final]
    data1['Category'] = cols_in_data_final 
    
    if criteria == 'Doc2Vec':
        from gensim.models.doc2vec import Doc2Vec, TaggedDocument
        from nltk.tokenize import word_tokenize
        data = data1['Text'].astype(str).tolist()

        tagged_data = [TaggedDocument(words=word_tokenize(_d.lower()), tags=[str(i)]) for i, _d in enumerate(data)]

        vec_size = 100
        ALPHA = 0.01
        max_epochs = 100
        DM = 1
        MIN_COUNT = 5
        MIN_ALPHA = 0.0001

        model = Doc2Vec(size=vec_size,
                    alpha=ALPHA, 
                    min_alpha=MIN_ALPHA,
                    min_count=MIN_COUNT,
                    dm =DM,
                    seed = 1234,
                    workers=1)

        model.build_vocab(tagged_data)

        for epoch in range(int(max_epochs)):
            #print('iteration {0}'.format(epoch))
            model.train(tagged_data,
                        total_examples=model.corpus_count,
                        epochs=model.iter)
            # decrease the learning rate
            model.alpha -= 0.0002
            # fix the learning rate, no decay
            model.min_alpha = model.alpha


        freq_count = data1['Category'].value_counts()
        freq_count = pd.DataFrame(freq_count)
        freq_count['catName'] = freq_count.index
        freq_count.columns = ['Cnt','catName']

        new_categories_to_fill=[]
        MAX_CAT = MAX_CAT

        j = 1
        while len(np.unique(data1['Category'])) > MAX_CAT and j < Max_Iter:
            new_categories_to_fill=[]
            for i in range(len(data1)):
                original_cat = data1.iloc[i]['Category']
                most_similar_cat = data1.iloc[int(model.docvecs.most_similar([i])[0][0])]['Category']
                orig_freq_cnt = freq_count[freq_count['catName']==original_cat]['Cnt'].values[0]
                most_similar_freq_cnt = freq_count[freq_count['catName']==most_similar_cat]['Cnt'].values[0]
                if orig_freq_cnt >= most_similar_freq_cnt:
                    new_categories_to_fill.append(original_cat)
                else:    
                    new_categories_to_fill.append(most_similar_cat)
            data1['Category'] = new_categories_to_fill
            freq_count = data1['Category'].value_counts()
            freq_count = pd.DataFrame(freq_count)
            freq_count['catName'] = freq_count.index
            freq_count.columns = ['Cnt','catName']

            j += 1
            print('Iteration....',j)
        
    if criteria == 'TfIdf':
        
        from sklearn.feature_extraction.text import TfidfVectorizer
        from scipy.sparse.csr import csr_matrix
        from nltk.corpus import stopwords
        from nltk.tokenize import word_tokenize
        from nltk.stem import WordNetLemmatizer
        import re
        from sklearn.metrics.pairwise import linear_kernel
        
        tf = TfidfVectorizer(input=data1['Text'].tolist(), analyzer='word', lowercase=False,ngram_range=(1,10),sublinear_tf=True,norm='l2')
        tfidf_matrix =  tf.fit_transform(data1['Text'].tolist())
        cosine_similarities = linear_kernel(tfidf_matrix,tfidf_matrix)
        np.fill_diagonal(cosine_similarities,-1)
        distances = np.argmax(cosine_similarities,axis=1)
        
        freq_count = data1['Category'].value_counts()
        freq_count = pd.DataFrame(freq_count)
        freq_count['catName'] = freq_count.index
        freq_count.columns = ['Cnt','catName']

        new_categories_to_fill=[]
        MAX_CAT = MAX_CAT
        
        j = 1
        while len(np.unique(data1['Category'])) > MAX_CAT and j < Max_Iter:
            new_categories_to_fill=[]
            for i in range(len(data1)):
                original_cat = data1.iloc[i]['Category']
                most_similar_cat = data1.iloc[distances[i]]['Category']
                orig_freq_cnt = freq_count[freq_count['catName']==original_cat]['Cnt'].values[0]
                most_similar_freq_cnt = freq_count[freq_count['catName']==most_similar_cat]['Cnt'].values[0]
                if orig_freq_cnt >= most_similar_freq_cnt:
                    new_categories_to_fill.append(original_cat)
                else:    
                    new_categories_to_fill.append(most_similar_cat)
            data1['Category'] = new_categories_to_fill
            freq_count = data1['Category'].value_counts()
            freq_count = pd.DataFrame(freq_count)
            freq_count['catName'] = freq_count.index
            freq_count.columns = ['Cnt','catName']

            j += 1
            print('Iteration....',j)
            
    if criteria == 'Word2Vec-PretrainedGoogle':
        import gensim.models.keyedvectors as word2vec
        model = word2vec.KeyedVectors.load_word2vec_format('C:/Users/antutlan/Microsoft/DSSInsightsAA - OPS-NWS Leakage/SACaseNotesExperimentation/GoogleNews-vectors-negative300.bin.gz', binary=True)
        
        from scipy import spatial

        index2word_set = set(model.wv.index2word)

        def avg_feature_vector(sentence, model, num_features, index2word_set):
            words = sentence.split()
            feature_vec = np.zeros((num_features, ), dtype='float32')
            n_words = 0
            for word in words:
                if word in index2word_set:
                    n_words += 1
                    feature_vec = np.add(feature_vec, model[word])
            if (n_words > 0):
                feature_vec = np.divide(feature_vec, n_words)
            return feature_vec
        
        foo = avg_feature_vector(data1['Text'][0], model=model, num_features=300, index2word_set=index2word_set)
        foo = pd.DataFrame(foo).T
        fooToFill = foo[0:0]
        for i in range(len(data1)):
            foo = avg_feature_vector(data1['Text'][i], model=model, num_features=300, index2word_set=index2word_set)
            foo = pd.DataFrame(foo).T
            fooToFill = pd.concat([fooToFill,foo],axis=0)
        
        from sklearn.metrics.pairwise import cosine_distances
        cosine_similarities = 1 - cosine_distances(fooToFill,fooToFill)
        np.fill_diagonal(cosine_similarities,-1)
        distances = np.argmax(cosine_similarities,axis=1)
        
        freq_count = data1['Category'].value_counts()
        freq_count = pd.DataFrame(freq_count)
        freq_count['catName'] = freq_count.index
        freq_count.columns = ['Cnt','catName']

        new_categories_to_fill=[]
        MAX_CAT = MAX_CAT
        
        j = 1
        while len(np.unique(data1['Category'])) > MAX_CAT and j < Max_Iter:
            new_categories_to_fill=[]
            for i in range(len(data1)):
                original_cat = data1.iloc[i]['Category']
                most_similar_cat = data1.iloc[distances[i]]['Category']
                orig_freq_cnt = freq_count[freq_count['catName']==original_cat]['Cnt'].values[0]
                most_similar_freq_cnt = freq_count[freq_count['catName']==most_similar_cat]['Cnt'].values[0]
                if orig_freq_cnt >= most_similar_freq_cnt:
                    new_categories_to_fill.append(original_cat)
                else:    
                    new_categories_to_fill.append(most_similar_cat)
            data1['Category'] = new_categories_to_fill
            freq_count = data1['Category'].value_counts()
            freq_count = pd.DataFrame(freq_count)
            freq_count['catName'] = freq_count.index
            freq_count.columns = ['Cnt','catName']

            j += 1
            print('Iteration....',j)
    
    if criteria=='Word2Vec-Text8Corpus':
        from gensim.models.word2vec import Text8Corpus
        from gensim.models import Word2Vec
        w2v_model2 = Word2Vec(Text8Corpus('C:/Users/antutlan/Downloads/text8/text8'), size=100, window=5, min_count=5, workers=4)
        index2word_set = set(w2v_model2.wv.index2word)

        def avg_feature_vector_Text8(sentence, model, num_features, index2word_set):
            words = sentence.split()
            feature_vec = np.zeros((num_features, ), dtype='float32')
            n_words = 0
            for word in words:
                if word in index2word_set:
                    n_words += 1
                    feature_vec = np.add(feature_vec, model[word])
            if (n_words > 0):
                feature_vec = np.divide(feature_vec, n_words)
            return feature_vec
        
        foo = avg_feature_vector_Text8(data1['Text'][0], model=w2v_model2, num_features=100, index2word_set=index2word_set)
        foo = pd.DataFrame(foo).T
        fooToFill = foo[0:0]
        for i in range(len(data1)):
            foo = avg_feature_vector_Text8(data1['Text'][i], model=w2v_model2, num_features=100, index2word_set=index2word_set)
            foo = pd.DataFrame(foo).T
            fooToFill = pd.concat([fooToFill,foo],axis=0)
            
        from sklearn.metrics.pairwise import cosine_distances
        cosine_similarities = 1 - cosine_distances(fooToFill,fooToFill)
        np.fill_diagonal(cosine_similarities,-1)
        distances = np.argmax(cosine_similarities,axis=1)
        
        freq_count = data1['Category'].value_counts()
        freq_count = pd.DataFrame(freq_count)
        freq_count['catName'] = freq_count.index
        freq_count.columns = ['Cnt','catName']

        new_categories_to_fill=[]
        MAX_CAT = MAX_CAT
        
        j = 1
        while len(np.unique(data1['Category'])) > MAX_CAT and j < Max_Iter:
            new_categories_to_fill=[]
            for i in range(len(data1)):
                original_cat = data1.iloc[i]['Category']
                most_similar_cat = data1.iloc[distances[i]]['Category']
                orig_freq_cnt = freq_count[freq_count['catName']==original_cat]['Cnt'].values[0]
                most_similar_freq_cnt = freq_count[freq_count['catName']==most_similar_cat]['Cnt'].values[0]
                if orig_freq_cnt >= most_similar_freq_cnt:
                    new_categories_to_fill.append(original_cat)
                else:    
                    new_categories_to_fill.append(most_similar_cat)
            data1['Category'] = new_categories_to_fill
            freq_count = data1['Category'].value_counts()
            freq_count = pd.DataFrame(freq_count)
            freq_count['catName'] = freq_count.index
            freq_count.columns = ['Cnt','catName']

            j += 1
            print('Iteration....',j)
            
    data1['Text'] = text_to_be_used_later
    return data1

