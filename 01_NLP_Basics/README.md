### NLP Basics - Tokenization, Normalization, Stemming, Lemmatization, webpage extraction using BeautifulSoup

#### Getting used to the NLTK package.

A text of about 200 words was taken from ball-by-ball commentary on ESPNCricinfo of a recent Ashes match for the purpose of this practical. (ESPNCricinfo, 2019)

**Using NLTK's word tokenizer, and reporting oddities:** 

```
['Woakes', 'to', 'Smith', ',', 'OUT', ',', 'he', "'s", 'walked', 'across', 'his', 'stumps', 'and', 'Smith', 0'has', 'missed', 'a', 'straight', 'one', '!', 'STEVEN', 'SMITH', 'HAS', 'MISSED', 'A', 'STRAIGHT', 'ONE', '!', 'Oh', 'my', 'word', '...', 'a', 'perfectly', 'reasonable', 'off-stump', 'angled', 'delivery', ',', 'pretty', 'innocuous', 'if', 'truth', 'be', 'told', ',', 'but', 'it', 'did', 'little', 'enough', 'not', 'to', 'bounce', 'over', 'the', 'top', 'as', 'the', 'master', 'momentarily', 'muddled', 'his', 'calculations', '.', 'Would', 'have', 'crashed', 'into', 'middle', 'and', 'leg', ',', 'two-thirds', 'up', '.', 'And', 'well', ',', 'crikey', '.', 'Smith', 'is', 'L.B.W', 'and', 'this', 'might', 'be', 'England', "'s", 'golden', 'opportunity', '!', 'Archer', 'to', 'Lyon', ',', 'OUT', ',', 'beautiful', 'yorker', '!', 'Archer', 'reaches', 'into', 'his', 'O.D.I', 'box', 'of', 'tricks', ',', 'a', 'perfect', 'little', 'doodlebug', 'of', 'a', 'knuckleball', ',', '20mph', 'off', 'his', 'top', 'pace', ',', 'fingers', 'hidden', 'behind', 'the', 'seam', 'as', 'if', 'switching', 'off', 'the', 'motors', 'to', 'glide', 'to', 'its', 'target', ',', 'and', 'crashes', 'into', 'the', 'off', 'stump', ',', 'right', 'on', 'the', 'creaseline', ',', 'to', 'complete', 'a', 'richly', 'deserved', '5-for', '!', 'Lyon', 'had', 'no', 'answer', 'whatsoever', ',', 'but', 'that', 'was', 'a', 'superb', 'counterattacking', 'cameo', 'that', 'has', 'eaten', 'into', 'a', 'dangerous', 'deficit', '.', 'Archer', 'to', 'Siddle', ',', 'OUT', ',', 'outstanding', 'snaffle', 'from', 'Burns', 'in', 'the', 'gully', '!', 'That', 'is', 'stunning', ',', 'reminiscent', 'of', 'Stokes', 'at', 'Trent', 'Bridge', 'in', '2015', '!', 'Full', 'length', 'to', 'his', 'right', ',', 'stretching', 'so', 'low', 'that', 'he', 'was', 'almost', 'reaching', 'behind', 'him', 'by', 'the', 'time', 'it', 'reached', 'him.He', "'s", 'gone', '!']
```

***From the tokenised list, it can be seen that numbers (2015), abbreviations (L.B.W, O.D.I) and hyphenated words (off-stump, two-thirds, 5-for) are handled well by the tokenizer. However, there are also some oddities observed.***

| Oddities                                                                      | How to handle them                                                             |
| ----------------------------------------------------------------------------- | ------------------------------------------------------------------------------ |
| Words with an apostrophe are split (“He” and “‘s”, “England” and “‘s”)        | Might be expected, but convert words into two different root terms (He and is) |
| “him.He” is tokenised as one token                                            | Possibly an issue with the textual data.                                       |
| Punctuations (commas, quotes, dots, exclamation marks) are not handled at all | Separately process data to remove them from the text.                          |

####  Normalization

For normalization, stopwords were removed and all the tokens were converted to lowercase. The output of this process is:
```
['woakes', 'smith', ',', ',', "'s", 'walked', 'across', 'stumps', 'smith', 'missed', 'straight', 'one', '!', 'steven', 'smith', 'missed', 'straight', 'one', '!', 'oh', 'word', '...', 'perfectly', 'reasonable', 'off-stump', 'angled', 'delivery', ',', 'pretty', 'innocuous', 'truth', 'told', ',', 'little', 'enough', 'bounce', 'top', 'master', 'momentarily', 'muddled', 'calculations', '.', 'would', 'crashed', 'middle', 'leg', ',', 'two-thirds', '.', 'well', ',', 'crikey', '.', 'smith', 'l.b.w', 'might', 'england', "'s", 'golden', 'opportunity', '!', 'archer', 'lyon', ',', ',', 'beautiful', 'yorker', '!', 'archer', 'reaches', 'o.d.i', 'box', 'tricks', ',', 'perfect', 'little', 'doodlebug', 'knuckleball', ',', '20mph', 'top', 'pace', ',', 'fingers', 'hidden', 'behind', 'seam', 'switching', 'motors', 'glide', 'target', ',', 'crashes', 'stump', ',', 'right', 'creaseline', ',', 'complete', 'richly', 'deserved', '5-for', '!', 'lyon', 'answer', 'whatsoever', ',', 'superb', 'counterattacking', 'cameo', 'eaten', 'dangerous', 'deficit', '.', 'archer', 'siddle', ',', ',', 'outstanding', 'snaffle', 'burns', 'gully', '!', 'stunning', ',', 'reminiscent', 'stokes', 'trent', 'bridge', '2015', '!', 'full', 'length', 'right', ',', 'stretching', 'low', 'almost', 'reaching', 'behind', 'time', 'reached', 'him.he', "'s", 'gone', '!']
```
 
#### POS Tagging

After running through a pos-tagger, the output is as follows: 

``` 
[('woakes', 'NNS'), ('smith', 'NN'), (',', ','), (',', ','), ("'s", 'POS'),('walked', 'VBN'), ('across', 'IN'), ('stumps', 'NNS'), ('smith', 'VBP'), ('missed', 'VBN'), ('straight', 'JJ'), ('one', 'CD'), ('!', '.'), ('steven', 'NN'), ('smith', 'NN'), ('missed', 'VBD'), ('straight', 'JJ'), ('one', 'CD'), ('!', '.'), ('oh', 'NN'), ('word', 'NN'), ('...', ':'), ('perfectly', 'RB'), ('reasonable', 'JJ'), ('off-stump', 'NN'), ('angled', 'JJ'), ('delivery', 'NN'), (',', ','), ('pretty', 'RB'), ('innocuous', 'JJ'), ('truth', 'NN'), ('told', 'VBD'), (',', ','), ('little', 'JJ'), ('enough', 'JJ'), ('bounce', 'NN'), ('top', 'JJ'), ('master', 'NN'), ('momentarily', 'RB'), ('muddled', 'VBD'), ('calculations', 'NNS'), ('.', '.'), ('would', 'MD'), ('crashed', 'VB'), ('middle', 'JJ'), ('leg', 'NN'), (',', ','), ('two-thirds', 'NNS'), ('.', '.'), ('well', 'RB'), (',', ','), ('crikey', 'NN'), ('.', '.'), ('smith', 'NN'), ('l.b.w', 'NN'), ('might', 'MD'), ('england', 'VB'), ("'s", 'POS'), ('golden', 'JJ'), ('opportunity', 'NN'), ('!', '.'), ('archer', 'RB'), ('lyon', 'NN'), (',', ','), (',', ','), ('beautiful', 'JJ'), ('yorker', 'NN'), ('!', '.'), ('archer', 'NN'), ('reaches', 'VBZ'), ('o.d.i', 'JJ'), ('box', 'NN'), ('tricks', 'NNS'), (',', ','), ('perfect', 'JJ'), ('little', 'JJ'), ('doodlebug', 'NN'), ('knuckleball', 'NN'), (',', ','), ('20mph', 'CD'), ('top', 'JJ'), ('pace', 'NN'), (',', ','), ('fingers', 'NNS'), ('hidden', 'VBP'), ('behind', 'IN'), ('seam', 'NN'), ('switching', 'VBG'), ('motors', 'NNS'), ('glide', 'RB'), ('target', 'NN'), (',', ','), ('crashes', 'NNS'), ('stump', 'VBP'), (',', ','), ('right', 'JJ'), ('creaseline', 'NN'), (',', ','), ('complete', 'JJ'), ('richly', 'RB'), ('deserved', 'VBN'), ('5-for', 'JJ'), ('!', '.'), ('lyon', 'NN'), ('answer', 'NN'), ('whatsoever', 'NN'), (',', ','), ('superb', 'NN'), ('counterattacking', 'VBG'), ('cameo', 'NNS'), ('eaten', 'RB'), ('dangerous', 'JJ'), ('deficit', 'NN'), ('.', '.'), ('archer', 'CC'), ('siddle', 'JJ'), (',', ','), (',', ','), ('outstanding', 'JJ'), ('snaffle', 'NN'), ('burns', 'NNS'), ('gully', 'RB'), ('!', '.'), ('stunning', 'NN'), (',', ','), ('reminiscent', 'JJ'), ('stokes', 'NNS'), ('trent', 'NN'), ('bridge', 'NN'), ('2015', 'CD'), ('!', '.'), ('full', 'JJ'), ('length', 'NN'), ('right', 'NN'), (',', ','), ('stretching', 'VBG'), ('low', 'JJ'), ('almost', 'RB'), ('reaching', 'VBG'), ('behind', 'IN'), ('time', 'NN'), ('reached', 'VBD'), ('him.he', 'NN'), ("'s", 'POS'), ('gone', 'VBN'), ('!', '.')]
```

At this stage, POS tagger outputs a tuple with the word, and the tag generated. Quite a lot of inaccuracies can be seen here, some of which are:

* Proper nouns are not handled well all the time. “Archer”, “Smith”, "England" and "Siddle" are tagged as RB, VBP, VB	    and JJ respectively at various stages.
* “Crikey” and "oh" should be interjections (UH), and not a noun (NN)
* A couple of cases where words are incorrectly tagged as adverbs (gully which is a noun, eaten whichis a verb)
* The adjectives "stunning" and "superb" are tagged as Nouns. (NN), while "counterattacking" is tagged as a Verb		    (VBG)
* "Crashes" and "bounce" are used as verbs, but tagged as nouns. 
* The abbreviation "o.d.i" has been incorrectly tagged as an adverb.
* "Deserved" (VBN, should be adjective) and "whatsoever"(NN, should be adverb) are both incorrectly tagged.

#### Tokenizing a new text file, and using Porter Stemming

A new text file was generated using an article on a new climate report on space.com. (Gohd, 2019)
On using Porter Stemming, the output is as follows:

```
['yesterday', '(', 'sept.', '25', ')', ',', 'a', 'stark', 'new', 'climat', 'report', 'came', 'out', 'that', 'show', 'climat', 'chang', 'is', 'progress', 'much', 'faster', 'than', 'anticip', ',', 'but', 'it', "'s", 'not', 'too', 'late', 'for', 'human', 'to', 'make', 'chang', '.', 'the', 'report', ',', 'written', 'and', 'releas', 'by', 'the', 'unit', 'nations-l', 'intergovernment', 'panel', 'on', 'climat', 'chang', '(', 'ipcc', ')', ',', 'and']
```

Porter stemmer brutally handles stemming. In this example, it has done so by: 
* Converting ending y to i (already - alreadi, policy - polici) 
* Cutting off -ing and -ed (released - releas, including - includ)
* Cutting off trailing e (change - chang, climate - climat, cryosphere - cryospher, dioxide - dioxid, continue - continu)
* Dealing with plurals by removing trailing s or es (changes - chang, contributes - contribut, impacts - impact)
* Also deals with suffixes like -ion, -al and -ation.

#### Lemmatization:

The output is as follows:

```
[('Yesterday', 'n'), ('(', 'n'), ('Sept.', 'n'), ('25', 'n'), (')', 'n'), (',', 'n'), ('a', 'n'), ('stark', 'a'), ('new', 'a'), ('climate', 'n'), ('report', 'n'), ('come', 'v'), ('out', 'r'), ('that', 'n'), ('show', 'v'), ('climate', 'a'), ('change', 'n'), ('be', 'v'), ('progress', 'v'), ('much', 'r'), ('faster', 'r'), ('than', 'n'), ('anticipate', 'v'), (',', 'n'), ('but', 'n'), ('it', 'n'), ("'s", 'v'), ('not', 'r'), ('too', 'r'), ('late', 'a'), ('for', 'n'), ('human', 'n'), ('to', 'n'), ('make', 'v'), ('change', 'n'), ('.', 'n'), ('The', 'n'), ('report', 'n'), (',', 'n'), ('write', 'v'), ('and', 'n'), ('release', 'v'), ('by', 'n'), ('the', 'n'), ('United', 'n'), ('Nations-led', 'a'), ('Intergovernmental', 'n'), ('Panel', 'n'), ('on', 'n'), ('Climate', 'n'), ('Change', 'n'), ('(', 'n'), ('IPCC', 'n'), (')', 'n'), (',', 'n'), ('and', 'n')]
```

The WordNet lemmatizer gives us the lemmas of words, their root form, but only if the root form is in the dictionary, unlike stemming. In the given text, it handles plurals (implications - implication, oceans - ocean) and deals with a  few suffixes (ing, ed - including to include, discussed to discuss, anticipated to anticipate) but little else.
Some of the things that look wrong are:

* as is converted to just ‘a’ after lemmatizing.
* us is converted to just ‘u’ after lemmatizing.
* Climate is tagged as an adjective during POS tagging.
  
We convert the pos_tags returned by pos_tagger to WordNet POS tags:


#### On comparing the outputs for Porter Stemming and Lemmatisation, we see quite a few differences.

Porter Stemmer is fast, and brutally changes just about every word according to its algorithm - converting trailing y to i,  cutting off -ing and -ed, etc. It also deals well with plurals, but the resulting words might not make sense.
On the other hand, WordNet Lemmatizer is slower, and returns lemmas of the word. But it only chops or changes the word if the resulting word is present in the dictionary. So it is more accurate. The Lemmatizer also needs POS tags for more accuracy.
So for systems where context is important, and we need to be accurate, lemmatization is better. However, for faster and crude text analysis, stemming would work fine.

Some of the words returned by the Stemmer and Lemmatizer for comparison:

| Original Word | Porter Stemmer | WordNet Lemmatizer |
| ------------- | -------------- | ------------------ |
| Turning       | Turn           | Turn               |
| Implications  | Implic         | Implication        |
| Policy        | Polici         | Policy             |
| Emissions     | Emiss          | Emission           |
| Released      | Releas         | Release            |


#### Parsing a remote webpage using BeautifulSoup

*A webpage of ball-by-ball commentary on ESPNCricinfo of a recent match between India and South Africa was used for extraction of text content. (ESPNCricinfo, 2019)
A portion of the entire webpage (soup):*
<!DOCTYPE html>
<html class="no-icon-fonts" lang="en">
<head>
<meta content="text/html; charset=utf-8" http-equiv="content-type"/>
<meta content="IE=edge,chrome=1" http-equiv="x-ua-compatible"/>
<meta content="initial-scale=1.0, maximum-scale=1.0, user-scalable=no" name="viewport"/>
<meta content="origin-when-cross-origin" name="referrer"/>
<link href="https://www.espncricinfo.com/series/19315/commentary/1187006/india-vs-south-africa-3rd-t20i-south-africa-in-india-2019-20" rel="canonical"/>
<title>Get Ball by Ball Commentary of India vs South Africa 3rd T20I 2019 | ESPNcricinfo.com</title>
<meta content="Read Commentary &amp; Updates of India vs South Africa 3rd T20I 2019, South Africa tour of India 2019/20 Cricket Match only on ESPNcricinfo.com. Check full commentary of India vs South Africa 3rd T20I" name="description"/>
<link href="https://www.espncricinfo.com/manifest.json" rel="manifest"/>

*Result of print(soup.head.title.text):*
Get Ball by Ball Commentary of India vs South Africa 3rd T20I 2019 | ESPNcricinfo.com

*Portion of the result of print(soup.body):*
<body class="gamepackage desktop prod scoreboard-fixed" data-app="" data-customstylesheet="null" data-edition="en-ww" data-lang="en" data-pagetype="gamepackage" data-site="espncricinfo" data-sport="cricket">
<div class="ad-slot ad-slot-exclusions" data-category-exclusion="true" data-slot-kvps="pos=exclusions" data-slot-type="exclusions"></div><div class="ad-slot ad-slot-overlay" data-out-of-page="true" data-slot-kvps="pos=outofpage" data-slot-type="overlay"></div><div class="ad-slot ad-slot-wallpaper" data-collapse-before-load="true" data-exclude-bp="s,m" data-slot-kvps="pos=wallpaper" data-slot-type="wallpaper"></div>

*Result of soup.find('body').find('p').getText():*
"10.32 pm: Well that ends the T20I series leg. It's been great fun bringing it to you. Will be even more fun when the Test series rolls around I'm sure, so make sure you join us for that one. Till then, this is Saurabh signing off on behalf of Karthik and Chandan. "

#### REFERENCES

ESPNCricinfo (2019). Get Ball by Ball Commentary of England vs Australia, ICC World Test Championship, 5th Test | ESPNcricinfo.com. Available at: https://www.espncricinfo.com/series/19430/commentary/1152850/england-vs-australia-5th-test-icc-world-test-championship-2019-2021?innings=2&filter=wickets 

Gohd, C. (2019). New Climate Report is Sobering but Strangely Hopeful. Available at: https://www.space.com/ipcc-2019-climate-report-sobering-hopeful.html

ESPNCricinfo (2019). Get Ball by Ball Commentary of India vs South Africa 3rd T20I 2019 | ESPNcricinfo.com
Available at:
https://www.espncricinfo.com/series/19315/commentary/1187006/india-vs-south-africa-3rd-t20i-south-africa-in-india-2019-20
