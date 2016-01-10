#import requests
import BeautifulSoup as bs
import re
import datetime as dt
from os import listdir
from os.path import isfile, join
#import nltk
import nameparser as parser
import dan
import encodings
from itertools import chain
import pickle
import networkx as nx
import itertools as it

###########################################
## uses "scrapeCaps" and "parseCaps" tools (tested elsewhere), then builds

# Multigraph
MG = nx.MultiGraph()

# General Graph
GG = nx.Graph()


########################################################################
indNum = re.compile(r'\d')
def scrapeCaps(soup, date):
# Given a BeautifulSoup object, parse it for valid captions 
# Returns a list of all valid captions for further work

	captions = []
	missed = False
	ppic = re.compile('/i/partypictures/')

	imgs = soup.findAll('img')
	for img in imgs:
		attrs = img.attrs

		for attr in attrs:
			if attr[0] == 'src':
				attr[1]
				src = attr[1]

		# if src=/i/partypics
		if ppic.match(src):

			try:
				## crawl up to /table level
				tst = img
				tab = False
				while not tab:
					name = tst.name
					if name != 'table':
						tst = tst.parent
					else:
						tab = tst
			# a handful of images aren't in named elements.  This skips them
			except AttributeError:
				break

			# work through table's children to find text
			tds = tab.findChildren()
			Ntd = 0
			cap = ''
			
			for ind, td in enumerate(tds):
				if td.text:

					if Ntd == 0:
						Ntd += 1
						first = td.text
						cap = cap + td.text

					# if there's more than one text field (often a repeat)
					else:
						if td.text != cap:

							# might be a substring
							if td.text not in cap:
								tcap = cap + td.text
								num = indNum.search(tcap)

								# substrings usually have CAPTION2, so split at \d
								if num:
									cap = tcap[ num.end():]

								# too much work to split these, only lose around 15
								else:
									missed = True

			# append if a cap was found
			if cap:
				captions.append(cap)

	## ^^ ABOVE COVERS VAST MAJORITY OF CASES ^^

	#  A handful of pages have the captions in alternate tables.  This tries another method
	if len(captions) == 0:

		switchdate = dt.date( 2007, 9, 4 )

		##  later, easier method of finding captions
		if date > switchdate:
			print " >> DIV HUNT << "

			divs = soup.findAll('div', 'photocaption')
			for div in divs :

				conts = div.contents
				# if this is a caption, parse it
				if conts:
					# make sure this is a string, or parseCaption crashes
					try:
						if conts[0].string:
							captions.append(conts[0].string)
					except TypeError:
						print "that wasn't a NavigableString"

		## trying a hunt for td labelled with photocaption
		if len(captions) == 0:
			print "  >> EARLY TIMES << "
			tds = soup.findAll('td', 'photocaption')
			for td in tds :

				conts = td.contents
				# if this is a caption, parse it
				if conts:
					# make sure this is a string, or parseCaption crashes
					try:
						if conts[0].string:
							captions.append(conts[0].string)
					except TypeError:
						print "that wasn't a NavigableString"


	## final warning if nothing found
	if len(captions)==0:
		print " NONE FOUND"
		dan.danpause()

	return captions, missed


#########################################################################
fand = re.compile('^ and ')
findAt = re.compile(' at ')
fwith = re.compile(' with ')
ffrnd = re.compile(r'\b[Ff]riend')
fjr = re.compile(r'\bJr\b')
fsr = re.compile(r'\bSr\b')
fdr = re.compile(r'\b[dD]\.?[rR]\.\b')	# matches Dr D.R. DR dr. 
fthe = re.compile(r'\b[Tt]he ')

#
fpunkt = re.compile(r"""
	(
		[\,]
	|	[\(\)]+		#
	|	[\:\!]+	# 
	|	"'s "+		# split on 's
	|	[\b-\b]+	# non-word hypens
	|	\s*$[.]+	# periods at start of chunk
)
""", re.VERBOSE)


def parseCaption(caption):
# given a caption, return a list of names of people in the caption
# if no names found, or caption not about people, return None
	debug = False
	flag = False
	rejects = []

	if debug:
		print " "
		print caption

	names = []
	chunks = []

	caplen = len(caption)
	if (caplen>1) & (caplen<250) :	# ignore short/long descriptions


		## split on 'AT' and 'IN'
		# throw away everything after "so-and-so AT the ball"
		if findAt.search(caption):
			rejects.append( caption[ findAt.search(caption).start(): ])
			caption = caption[ :findAt.search(caption).start() ].strip()
		if ' in ' in caption:
			tmp = caption.split(' in ')
			rejects.append( tmp[1] )
			caption = tmp[0]
		if debug:
			print "After at/in : ", caption



		### skip if less than 4 words
		if len( caption.split() ) < 4:
			if debug:
				print " nWords < 4"
			return None


		# each chunk is a set of words
		chunks = fpunkt.split(caption)


		###  fix Jr. / Sr. / Dr.
		for ind, chunk in enumerate(chunks):
			if fsr.search(chunk):
				print chunk
				spl = fjr.split(chunk)
				chunk = ''.join(spl).strip()
				print chunk

		for ind, chunk in enumerate(chunks):
			if fjr.search(chunk):
	#			print
				print chunk
				spl = fjr.split(chunk)
				chunk = ''.join(spl).strip()
				print chunk

		for ind, chunk in enumerate(chunks):
			if fdr.search(chunk):
				print
				print "Dr. Sub"
				print chunk
				spl = fdr.split(chunk)
				chunks[ind] = ''.join(spl).strip()
				print chunks
		## ^^ 	only the Dr. one works, because of Python's scoping.
		## 		Could fix the others if needed


		if debug:
			print chunks



		### split at "Bob WITH Kate"
		for ind,chunk in enumerate(chunks):
#				if 'with' in chunk:
			if fwith.search(chunk):
				pieces = chunk.split(' with ')
				chunks[ind] = pieces[0]
				chunks.insert(ind+1, pieces[1])





		### handling 'AND' ###

		# checks first word of split (implies a list, and Bob)
		for ind,chunk in enumerate(chunks):
			if fand.match(chunk):	# this re. defined above
				chunks[ind] = chunks[ind][5:]	# everything after the "and "
				chunk = chunks[ind]
				#print "post-fand chunks : ", chunks

			# separating "Husb and Wife Smith", etc
			if ' and ' in chunk:
				#print "found AND "
				temp = chunk.split(' and ')
				name1 = temp[0].strip()
				name2 = temp[1].strip()
				human1 = parser.HumanName(name1)
				human2 = parser.HumanName(name2)

				if debug:
					print "ind = ", ind, "; temp = ", temp
					print "name1 = ", name1, ";  name2 = ", name2
					print "human1 = ", human1
					print "human2 = ", human2

				# if this was of the form "Husband and Wife Smith"
				if not human1.last :
					human1 = temp[0].strip() + ' ' + human2.last
				else :
					human1 = name1

				chunks[ind]= human1
				chunks.insert(ind+1, temp[-1])




		### check for capitalized words to see if this is names ###
		cutList = []
		for ind, chunk in enumerate(chunks) :
			words = chunk.split()
			nWords = len(words)

			if nWords:
				# check ratio of caps to not
				nCaps = sum(map(str.isupper, str(chunk)))
#				ratio = float(nCaps)/nWords

				if (nWords-nCaps)>1 :
					cutList.append( chunk )
					if debug:
						print "no caps"

		rejects.append(cutList)
		for cut in cutList:
			chunks.remove( cut )

		if debug:
			print
			print "after capitals :"
			print "  reject : ", rejects
			print "  keep :", chunks
			dan.danpause()




		### cut chunks with 'The'
		cutList = []
		for chunk in chunks:
			if fthe.search( chunk ):
		#		print chunks
				# should probably check if it's already there
				if chunk not in cutList:
					cutList.append( chunk )

		rejects.append(cutList)
		for cut in cutList:
			chunks.remove( cut )

		if debug:
			print ' cutting "the" : ', chunks




		### upon exit ###
		if len(chunks)>1:		# need more than one person
			for chunk in chunks:
				chunk = chunk.strip()
				if len(chunk.split())>1:
					# make sure it doesn't say "friend"
					if not ffrnd.search(chunk):
						# strip whitespace, condense multispaces
						names.append( re.sub('\s+',' ',chunk.strip() ) )
					else :
						print "rej : ", chunk
						rejects.append(chunk)


	#	print names
		return names

	else:
		return None

##############################################################################
# saved/2013-05-22-spring-gala-galore.html.ascii
# 2011-01-13-history-buffs.html.ascii

folder = 'saved/'
mypath = folder #+ '*.ascii'

# this case is used for debugging individual files that are crashing
if 0:
	filename = '2010-07-08-independence-and-verve.html.ascii'
	filename = '2011-01-13-history-buffs.html.ascii'
#	filename = '2007-08-21-king-of-california-in-the-hamptons.html.ascii'
	# load file
	f = open(folder+filename,'r')
	resp = f.read()
	soup = bs.BeautifulSoup(resp)
	f.close()

	print filename
	captions,missed = scrapeCaps(soup, filename)
	
	## now parse them
	for cap in captions:
		nameLinks = parseCaption( cap )
		if nameLinks:
			print nameLinks


else:
	## goes through all the files in the folder, parses them, adds to multigraph

	onlyfiles = [ f for f in listdir(mypath) if isfile(join(mypath,f)) ]

	len(onlyfiles)

	iFile = 0
	allCaps = []
	nCaps = 0
	nMiss = 0
	nEmpty = 0
	for iFile, ofile in enumerate(onlyfiles):

		if iFile > -1:
			# determine date from filename
			datestr = ofile[0:10]
			yr = int(datestr[0:4])
			mon = int(datestr[5:7])
			day = int(datestr[8:10] )
			date = dt.date( yr, mon, day)

			# load file
			filename = folder + ofile
			print filename
			with open(filename,'r') as f:
				resp = f.read()
				clean = encodings.utf_8.decode( resp )[0]
				soup = bs.BeautifulSoup( clean )
				f.close()


			### return the scraped caps ###
			captions,missed = scrapeCaps(soup, date)

			## now parse them
			for cap in captions:
				nameLinks = parseCaption( cap )
				if nameLinks:
					for itr in it.combinations(nameLinks,2):
#						print itr
						MG.add_edge( itr[0], itr[1] )
#					print nameLinks

#		dan.danpause()


## after full multigraph is constructed, convert it to standard graph
for n,nbrs in MG.adjacency_iter():
	for nbr, edict in nbrs.items():
		GG.add_edge(n, nbr, weight = len(edict) )

## turns out there was an easier way to do this, but it worked
