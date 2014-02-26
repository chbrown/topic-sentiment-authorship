def links_scikit():
    # index_range = range(len(dictionary.index2token))
    # docmatrix = np.matrix([[bag_of_tfidf.get(index, 0) for index in index_range] for bag_of_tfidf in tfidf_documents])
    # for i, bag_of_tfidf in enumerate(tfidf_documents):
    #     index_scores = sorted(bag_of_tfidf.items(), key=lambda x: -x[1])
    #     doc = ['%s=%0.4f' % (dictionary.index2token[index], score) for index, score in index_scores]
    #     print i, '\t'.join(doc)

    # U, s, V = np.linalg.svd(docmatrix, full_matrices=False)
    # print docmatrix.shape, '->', U.shape, s.shape, V.shape
    maxlen = 10000

    import tsa.data.sb5b.links
    endpoints = tsa.data.sb5b.links.read(limit=1000)
    corpus_strings = (endpoint.content[:maxlen] for endpoint in endpoints)

    count_vectorizer = CountVectorizer(min_df=2, max_df=0.95)
    counts = count_vectorizer.fit_transform(corpus_strings)
    # count_vectorizer.vocabulary_ is a dict from strings to ints

    tfidf_transformer = TfidfTransformer()
    tfidf_counts = tfidf_transformer.fit_transform(counts)

    # count_vectors_index_tokens is a list (map from ints to strings)
    count_vectors_index_tokens = count_vectorizer.get_feature_names()

    # eg., 1000 documents, 2118-long vocabulary (with brown from the top)
    print count_vectors_index_tokens.shape

    pca = decomposition.PCA(2)
    doc_pca_data = pca.fit_transform(tfidf_counts.toarray())

    print doc_pca_data

    # target_vocab = ['man', 'men', 'woman', 'women', 'dog', 'cat', 'today', 'yesterday']
    # for token_type_i, coords in enumerate(vocab_pca_data):
    #     token = vector_tokens[token_type_i]
    #     if token in target_vocab:
    #         x, y = coords
    #         print "%s,%.12f,%.12f" % (token, x, y)
