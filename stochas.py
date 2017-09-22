import io
import numpy as np
from conllu.parser import parse
import os
import itertools
from func_PerformanceMetric import calc_PosNegValue,calc_Acc,calc_Prec,calc_Recall,calc_FScore

#####INIT DATA###########
def readFile(txtFile):
	with open(txtFile,encoding='utf8') as myfile:
		return myfile.read().lower()

def parse_sentence_conllu(conllu_raw_string):
#list of POS access: parsed[id_sent][id_word][x] x=0-> form,x = 1=pos
	conllu_list_raw = conllu_raw_string.replace("\t","   ").split("\n\n")
	conllu_list_final = []

	for clr in conllu_list_raw:
		if len(clr)>2:
			clr_part = clr.split("\n")
			conllu_final= ""
			for cline in range(2,len(clr_part)):
				conllu_final += clr_part[cline] +"\n"
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
		if f=='aux_list':
			pos_list['aux']=word_list
		else:
			pos_list[f]= word_list

######COUNT DATA#########
def tag_loc_in_sentence(parsed_sentence,tag):
	tag_loc = [word[1].lower() for word in parsed_sentence]
	return [i for i, s in enumerate(tag_loc) if tag==s]

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

def count_tag_prob(parsed,el_id,elmt):
	#el_id = 0-> word, 1->tag
	elmt_count = 0
	corpus_len = 0
	for parsed_sentence in parsed:
		corpus_len+= len(parsed_sentence)
		elmt_list = [word[el_id].lower() for word in parsed_sentence]
		elmt_count += elmt_list.count(elmt.lower())
	return elmt_count/corpus_len

#======= DATA INIT =========
pos_list = {}
tag_pair_t1 = []
tag_pair_t2 = []
tag_pair_count = []

tag_prob_tag = []
tag_prob_value = []

all_tag_pair = 0 
parsed = parse_sentence_conllu(readFile('UD_Indonesian/id-ud-train.conllu'))
parsed_dev = parse_sentence_conllu(readFile('UD_Indonesian/id-ud-dev.conllu'))
init_pos_list('tag')
for i in pos_list:
	tag_prob_tag.append(i)
	tag_prob_value.append(count_tag_prob(parsed,1,i))
	for y in pos_list:
		tag_pair_t1.append(i)
		tag_pair_t2.append(y)
		a = count_tag_pair_corpus(parsed,i,y)
		tag_pair_count.append(a)
		all_tag_pair += a

combine_parsed = []
for parsed_s in parsed:
	for parsed_w in parsed_s:
		combine_parsed.append(parsed_w[0].lower()+"_"+str(parsed_w[1].lower()))
print(len(combine_parsed))
#==========================

def count_tag_pair_corpus2(tag1,tag2):
	idx = tag_pair_t1.index(tag1)
	c = -1
	while (idx<len(tag_pair_t1) and tag_pair_t1[idx]==tag1 and c==-1):
		if tag_pair_t2[idx]==tag2:
			c = tag_pair_count[idx]
		else:
			idx+=1
	return c

print(count_tag_pair_corpus2('noun','noun'))

def listing_word_tags(word):
	tag_list = []
	for i in pos_list:
		if word in pos_list[i]:
			tag_list.append(i)
	return tag_list

def listing_sentc_tag_seq(sentence):
	tags_list = []
	words = sentence.lower().split(" ")
	for word in words:
		tags_list.append(listing_word_tags(word))
	#print(tags_list)
	return list(itertools.product(*tags_list))

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

def prob_tag_and_word2(tag,word):
	return combine_parsed.count(word.lower()+"_"+tag.lower())/len(combine_parsed)

def prob_w_given_t(w,t):
	#P(w & t)/p(t)
	p_w_and_t = prob_tag_and_word2(t,w)
	p_tag = tag_prob_value[tag_prob_tag.index(t)]
	return p_w_and_t/p_tag,p_tag

def prob_w_given_t2(w,t):
	#P(w & t)/p(t)
	p_w_and_t = prob_tag_and_word2(t,w)
	p_tag = tag_prob_value[tag_prob_tag.index(t)]
	return p_w_and_t/p_tag

def prob_t_given_tbev(t,tbev):
	#P(t & t-1)/P(t-1)
	p_and_tags = count_tag_pair_corpus2(tbev,t)/all_tag_pair
	p_tbev = tag_prob_value[tag_prob_tag.index(tbev)] #count_tag_prob(parsed,1,tbev)
	return p_and_tags/p_tbev

def prob_all_t_given_tbev(parsed,tag_seq):#one sequence
	prob = 0
	for i in range (1, len(tag_seq)):
		prob+= prob_t_given_tbev(parsed,tag_seq[i],tag_seq[i-1])
	return prob


def prob_all_w_given_t(parsed,word_list,tag_list):#one sequence
	prob = 0
	for i in range (0,len(word_list)):
		prob+= prob_w_given_t(word_list[i],tag_list[i])
	return prob

def prob_all_wgt_tgt(tag_seq,word_seq):
	wt = 0
	tt = 0
	pt = 0
	for i in range (0,len(tag_seq)):
		print(word_seq[i])
		wgt,ct = prob_w_given_t(word_seq[i],tag_seq[i])
		wt+= wgt
		if i>0:
			tt+= prob_t_given_tbev(tag_seq[i],tag_seq[i-1])
	return wt,tt

#-------------HMM---------------
#def pod_tag_hmm_unigram_one_s(parsed,tag_seq,word_list):

def pos_tag_hmm_bigram_one_s(parsed,tag_seq,word_list):#one sentence
	tags_prob = []#
	for i in range (0,len(tag_seq)):
		wt,tt = prob_all_wgt_tgt(tag_seq[i],word_list)
		pt =  tag_prob_value[tag_prob_tag.index(tag_seq[i][0])]
		tags_prob.append(tt*wt*pt)

	return tags_prob

def pos_tag_hmm_bigram_set(parsed,sentences):#whole dataset
	tags_max_prob = []
	tags_final_seq = []
	for sent in sentences:
		tag_seq = listing_sentc_tag_seq(sent)
		word_list = sent.split(' ')
		tags_prob = pos_tag_hmm_bigram_one_s(parsed,tag_seq,word_list)

		id_max = tags_prob.index(max(tags_prob))
		tags_max_prob.append(tags_prob[id_max])
		tags_final_seq.append(tag_seq[id_max])
	return tags_final_seq,tags_max_prob

def count_performance(all_tag_pred,all_tag_ground):
	tp,tn,fp,fn = 0
	for i in pos_list:
		tp1,tn1,fp1,fn1 = calc_PosNegValue(all_tag_ground, all_tag_pred, i)
		tp+=tp1
		tn+=tn1
		fp+=fp1
		fn+=fn1
	return calc_FScore(tp,tn,fp,fn)

def pos_tag_hmm_unigram_one_s(sentence):
	tag_seq = []
	sentence = sentence.split(' ')
	i =0
	for word in sentence:
		print(word)
		possible_tag = listing_word_tags(word)
		print(possible_tag)
		prob = -1
		curr_tag = 'noun'
		if i==0:
			for tag in possible_tag:
				wt,t = prob_w_given_t(word,tag)
				if wt*t>prob:
					prob = wt*t
					curr_tag = tag
		else:
			for tag in possible_tag:
				wt,t = prob_w_given_t(word,tag)
				t =  prob_t_given_tbev(tag,tag_seq[-1])
				if wt*t>prob:
					prob = wt*t
					curr_tag = tag
		i+=1
		tag_seq.append(curr_tag)
	return tag_seq


def pos_tag_hmm_unigram_set(sentences):
	'''
	11490
	12612
	'''
	tag_seq_sentence = []
	for s in sentences:
		print(s)
		tag_seq_sentence.append(pos_tag_hmm_unigram_one_s(s))
	return tag_seq_sentence

def b_viterbi(word_seq):
	b_word_tag_list = []
	b_prob_list = [] #p(w|t)
	for word in word_seq:
		for tag in pos_list:
			b_word_tag_list.append(word+"_"+tag)
			b_prob_list.append(prob_w_given_t2(word,tag))
	return b_word_tag_list,b_prob_list

def viterbi_hmm(word_seq):#one sent

	b_wt,b_prob = b_viterbi(word_seq)
	viterbi = []
	backpointer = []

	tags = []
	for pos in pos_list:
		tags.append(pos)

	for state in pos_list:
		viterbi.append([tag_prob_value[tag_prob_tag.index(state)]*b_prob[b_wt.index(word_seq[0].lower()+"_"+state)]])
		backpointer.append([-1])

	for w in range(1,len(word_seq)):
		for t in pos_list:
			max_viterbi = -1
			max_viterbi_idx = -1
			idx_pos= 0
			for prev_t in pos_list:
				if b_prob[b_wt.index(word_seq[w]+"_"+t)]>0:
					score = viterbi[idx_pos][w-1]*prob_t_given_tbev(t,prev_t)*b_prob[b_wt.index(word_seq[w]+"_"+t)]
				else:
					score = -1
				if score>max_viterbi:
					max_viterbi = score
					max_viterbi_idx = tags.index(prev_t)
				idx_pos+=1
			t_idx = list(pos_list.keys()).index(t)
			viterbi[t_idx].append(max_viterbi)
			#print("=="+str(max_viterbi_idx))
			backpointer[t_idx].append(max_viterbi_idx)
		#print(word_seq[w]+"_"+str(max_viterbi_idx))

		#check if all 0
		idx_all_pos = 0
		while (idx_all_pos<len(pos_list) and viterbi[idx_all_pos][w]==-1):
			idx_all_pos+=1
		#if all 0
		if idx_all_pos==len(pos_list):
			possible_tag = listing_word_tags(word_seq[w])
			t_idx = 0
			for t in possible_tag:
				max_viterbi = -1
				max_viterbi_idx = tags.index('noun')
				idx_pos=0
				
				for prev_t in pos_list:
					score = viterbi[idx_pos][w-1]*prob_t_given_tbev(t,prev_t)

					if score>max_viterbi:
						max_viterbi = score
						max_viterbi_idx = tags.index(prev_t)
					idx_pos+=1

				viterbi[list(pos_list.keys()).index(t)][w]= max_viterbi
				backpointer[list(pos_list.keys()).index(t)][w]=max_viterbi_idx

				t_idx+=1

	#final track
	final_track = []
	final_tag = []
	max_prob = -1
	max_id = 0

	for t in range(0,len(pos_list)):
		if max_prob < viterbi[t][-1]:
			max_prob = viterbi[t][-1]
			max_id = backpointer[t][-1]
	#print(max_id)
	final_track.append(max_id) #id_track
	final_tag.append(tags[max_id]) #tag_track
	#print(final_track[-1])
	idx_w = len(viterbi[0])-2
	while (idx_w>0):
		#print(word_seq[idx_w]+"|")
		#print(final_track)
		final_track.append(backpointer[final_track[-1]][idx_w])
		#final_tag.append(tags[final_track[-1]])
		final_tag.insert(0,tags[final_track[-1]])
		idx_w-=1
	#print(backpointer)
	#print(final_tag)
	'''
		for w in range(0,len(viterbi[0])):
			max_prob = 0
			max_id = 0
			for t in range(0,len(pos_list)):
				if max_prob<viterbi[t][w]:
					max_prob = viterbi[t][w]
					max_id = backpointer[t][w]
			final_track.append(max_id)
	'''
	return final_tag #viterbi

def viterbi_hmm_set(sentences):
	'''
	10098
	12612
	'''
	tag_seq_sentence = []
	for s in sentences:
		print(s)
		tag_seq_sentence.append(viterbi_hmm(s.split(" ")))
	return tag_seq_sentence

sentence_dev = []
i=0

for dev_p in parsed_dev:
	i+=1
	sentence_dev.append("")
	for word in dev_p:
		sentence_dev[-1]+=word[0]+" "
	
'''
print("------------------")
print(sentence_dev)
print("------------------")

prob_w,ori = b_viterbi(sentence_dev[0].split(" "))
for i in range(0,len(ori)):
	print(prob_w[i]+" "+str(ori[i]))
'''
#print(viterbi_hmm(sentence_dev[0].split(" ")))
#tags,probs = pos_tag_hmm_bigram_set(parsed,sentence_dev[:1])
#print(sentence_dev[:1])

#tag = (pos_tag_hmm_unigram_set(sentence_dev))
tag = (viterbi_hmm_set(sentence_dev))
corr=0
alle=0
for i in range(0,len(tag)):
	for y in range (0,len(parsed_dev[i])):
		alle+=1
		if tag[i][y]==parsed_dev[i][y][1]:
			corr+=1
print(tag)
print(corr)
print(alle)


