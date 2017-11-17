# tsa: Topic-Sentiment Authorship

Tools built on computational linguistic concepts to accompany my research at the University of Texas at Austin.


## Development

    SITE_PACKAGES=$(python -c 'import os,site;print(os.path.realpath(next(iter(site.getsitepackages()))))')
    pwd > "$SITE_PACKAGES"/topic-sentiment-authorship.pth


## Results

`gensim` LDA, 10 topics, tf-idf preprocessing, on the 6,379 linked pages with content.

| Topic # | Top 20 tokens |
|:--------|:--------------|
| Topic 0 | jfs, teamsters, martin, jackson, foster, beavercreek, arrests, girls, demarcus, cincinnati, hayes, felony, you, our, org, teamster, sotheby, babies, issue, occupy |
| Topic 1 | blackberry, url, bryant, huffman, ride, supercommittee, transactions, thunder, bluetooth, rand, populism, frost, gd, annie, newer, brooks, revenge, unwillingness, patricia, rosen |
| Topic 2 | moveon, sumi, belmont, livestream, stewart, forefront, eagle, menu, copying, dalby, tavares, juvenile, charleta, deck, careful, sb, jimmy, you, your, please |
| Topic 3 | harris, signatures, teresa, marian, sb, marquis, id, said, ohio, bill, oregon, prisons, kasich, repeal, senate, state, we, teachers, law, nlrb |
| Topic 4 | thinkprogress, privacy, aol, hotmail, applicable, governed, clicking, yahoo, submitting, tags, terms, hagan, acknowledge, examiners, use, examiner, agree, facebook, policies, understand |
| Topic 5 | cwa, twitpic, dale, littleton, locating, io, eckart, mattei, clairsville, casey, oea, marshall, colleen, nbc, webpage, butland, robin, pew, hannah, innovation |
| Topic 6 | said, percent, law, bill, kasich, mr, ohio, he, union, voters, employees, senate, republican, signatures, repeal, ballot, issue, state, health, unions |
| Topic 7 | mobile, pearson, ali, repubs, alison, smooth, upgrade, frazier, species, rsvp, parma, pmwhere, ohwhat, gas, kasich, schiavoni, he, oil, your, you |
| Topic 8 | mississippi, biden, sexual, schuler, sep, weingarten, crackdown, charges, gears, midwest, pronounced, congratulated, setbacks, chicago, fertilized, reading, relating, changing, guilty, lgbt |
| Topic 9 | frey, school, tax, you, teachers, schools, she, education, bowl, my, your, income, her, philadelphia, york, states, business, teacher, unemployment, ohio |


## Notes

See `/usr/share/postgresql/tsearch_data/english.stop` for postgres' listing of stop words.


### Numpy slicing

From the docs:

> If the number of objects in the selection tuple is less than N,
> then : is assumed for any subsequent dimensions

Thus, if `coefs` is a two-dimensional array, `coefs[fold, ]` is the same as `coefs[fold, :]` (but not the same as `coefs[fold]`).


### Numpy axes:

**Aggregate functions**

* `axis=0`: apply function to each column in turn
* `axis=1`: apply function to each row in turn

When our rows are observations, **most aggregations use axis=0.**
This is because each cell has much more in common with the rest of the column than the rest of the row.

    >>> grades_by_age = np.array([
        [98, 14],
        [92, 15],
        [87, 13],
        [93, 14]])
    >>> grades_by_age.mean(axis=0)
    array([ 92.5,  14. ])
    >>> grades_by_age.mean(axis=1)
    array([ 56. ,  53.5,  50. ,  53.5])

But if you are selecting features, or labels, axis=1 is probably what you want.

This is also helpful: http://pages.physics.cornell.edu/~myers/teaching/ComputationalMethods/python/arrays.html


### Numpy array creation

Supposedly, specifying `count` speeds up `fromiter`:

    coefs = np.fromiter(bootstrap_coefs(folds), count=K)




### Numpy concatenation

- `hstack`: Stack arrays in sequence horizontally (column wise).
- `vstack`: Stack arrays in sequence vertically (row wise).


### Numpy broadcasting

Broadcasting just works with dense arrays, but fails miserably with sparse matrices.

    dense = np.arange(15).reshape(-1, 3)
    sparse = scipy.sparse.csr_matrix(dense)

    vec = np.array([2, 2, 2])
    (dense * vec).shape == (5, 3)
    (sparse * vec).shape == (5,)

WTF? `sparse * vec == sparse.dot(vec)`, which is not what I meant at all.


## License

Copyright © 2012–2014 Christopher Brown. [MIT Licensed](LICENSE).
