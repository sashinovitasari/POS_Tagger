import io
import numpy as np
from conllu.parser import parse
import os
import itertools

#####INIT DATA###########
def readFile(txtFile):
	with open(txtFile,encoding='utf8') as myfile:
		return myfile.read().lower()

def parse_sentence_conllu(conllu_raw_string):
#list of POS access: parsed[id_sent][id_word][x] x=0-> form,x = 1=pos
	conllu_list_raw = conllu_raw_string.split("\n\n")
	conllu_list_final = []

	for clr in conllu_list_raw:
		if len(clr)>2:
			clr_part = clr.split("\n")
			conllu_final= ""
			for cline in range(2,len(clr_part)):
				conllu_final += clr_part[cline].replace("\t","   ")+"\n"
			parsed = parse(conllu_final)

			conllu_list_final.append([])
			for idx in range(0,len(parsed[0])):
				par = []
				par.append(parsed[0][idx]['form'])
				par.append(parsed[0][idx]['upostag'])
				conllu_list_final[-1].append(par)
	return conllu_list_final

def init_pos_list(folder):
	files = os.listdir(folder)
	for f in files:
		word_list = readFile(folder+"/"+f).lower().split("\n")
		if word_list.count(' ')>0:
			word_list = word_list.remove('') 
		pos_list[f]= word_list

######COUNT DATA#########
def tag_loc_in_sentence(parsed_sentence,tag):
	tag_loc = []
	for id_word in range (0,len(parsed_sentence)):
		if (parsed_sentence[id_word][1]).lower()==tag.lower():
			tag_loc.append(id_word)
	return tag_loc

def count_tag_pair_sentence(parsed_sentence,tag1,tag2):
	tag1_loc = tag_loc_in_sentence(parsed_sentence,tag1)
	if tag1_loc!=[]:
		pair_count=0
		for id_tag1 in tag1_loc:
			if (id_tag1!=len(parsed_sentence)-1):
				if (parsed_sentence[id_tag1+1][1]).lower()==tag2.lower():
					pair_count+= 1
		return pair_count
	return 0 

def count_tag_pair_corpus(parsed,tag1,tag2):
	count_pair = 0
	for parsed_sentence in parsed:
		count_pair += count_tag_pair_sentence(parsed_sentence,tag1,tag2)
	return count_pair
#======= DATA INIT =========
pos_list = {}
all_tag_pair = 0 
parsed = parse_sentence_conllu(readFile('UD_Indonesian/id-ud-dev.conllu'))
init_pos_list('tag')
for i in pos_list:
	for y in pos_list:
		all_tag_pair += count_tag_pair_corpus(parsed,i,y)
#==========================

def listing_word_tags(word):
	tag_list = []
	for i in pos_list:
		if word in pos_list[i]:
			tag_list.append(i)
	return tag_list
	#print(tag_list)

def listing_sentc_tag_seq(sentence):
	tags_list = []
	words = sentence.lower().split(" ")
	for word in words:
		tags_list.append(listing_word_tags(word))

	return list(itertools.product(*tags_list))






#P(tag & tag+1)


def count_tag_prob(parsed,el_id,elmt):
	#el_id = 0-> word, 1->tag
	elmt_count = 0
	corpus_len = 0
	for parsed_sentence in parsed:
		corpus_len+= len(parsed_sentence)
		elmt_list = [word[el_id].lower() for word in parsed_sentence]
		elmt_count += elmt_list.count(elmt.lower())
	return elmt_count/corpus_len

def prob_tag_and_word(parsed,tag,word):
	corpus_len=0
	pair_count = 0
	for parsed_sentence in parsed:
		corpus_len+= len(parsed_sentence)
		id_words = [word[0].lower() for word in parsed_sentence]
		id_words = [i for i,val in enumerate(id_words) if val==word]
		for id_parsed_word in id_words:
			if parsed_sentence[id_parsed_word][1].lower()==tag.lower():
				pair_count += 1
	return pair_count/corpus_len

def prob_w_given_t(parsed,w,t):
	#P(w & t)/p(t)
	p_w_and_t = prob_tag_and_word(parsed,t,w)
	p_tag = count_tag_prob(parsed,1,t)
	return p_w_and_t/p_tag

def prob_t_given_tbev(parsed,t,tbev):
	#P(t & t-1)/P(t-1)
	c_and_tags = count_tag_pair_corpus(parsed,tbev,t)

	p_and_tags = c_and_tags/all_tag_pair
	p_tbev = count_tag_prob(parsed,1,tbev)
	return p_and_tags/p_tbev

def prob_all_t_given_tbev(parsed,tag_seq):#one sequence
	prob = 0
	for i in range (1, len(tag_seq)):
		prob+= prob_t_given_tbev(parsed,tag_seq[i],tag_seq[i-1])
	return prob

def prob_all_w_given_t(parsed,word_list,tag_list):#one sequence
	prob = 0
	for i in range (0,len(word_list)):
		prob+= prob_w_given_t(parsed,word_list[i],tag_list[i])
	return prob



 


#-------------HMM---------------
#def p_tag_given_word(parsed,tag,word):



#print(tag_loc_in_sentence(parsed[0],'propn'))
#print(count_tag_pair_sentence(parsed[0],'punct','noun'))
#print(parsed[-1][0][1])
#print(count_tag_prob(parsed,0,'saya'))
#print(count_tag_and_word_prob(parsed,'punct',','))



#listing_word_tags('yaitu')

tag_seq = (listing_sentc_tag_seq('Ahli rekayasa optik'))
word_list = ('Ahli rekayasa optik').split(' ')
'''
for tag in tag_seq:
	print(count_all_t_given_tbev(parsed,tag))
'''
prob_list = []
for i in range (0,len(tag_seq)):
	tt = (prob_all_t_given_tbev(parsed,tag_seq[i]))
	wt = prob_all_w_given_t(parsed,word_list,tag_seq[i])
	pt = count_tag_prob(parsed,1,tag_seq[i][0])
	print(tag_seq[i])
	print(tt*wt*pt)



