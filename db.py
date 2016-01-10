import pandas as pd
import sqlite3
from dan import danpause
import numpy as np

db = sqlite3.connect('nycql.db')
cur = db.cursor()

#####################################################################

def parseZgroup(zgroup):
# zip-code grouped data has tuples:
# (zipcode, number of restaurants, sum(score), sum(score**2) )
#
# this parses, computes mean, stdev, and then pretty-prints for copypasta
# to __init__.py

	zips = [ r[0] for r in zgroup ]
	nRst = np.array([ r[1] for r in zgroup ])
	score= np.array([ r[2] for r in zgroup ])
	scsqr= np.array([ r[3] for r in zgroup ])

	mean = score / nRst
	stds = np.sqrt( ((scsqr/nRst) - mean**2 )/ (nRst) )

	iSort = np.argsort( mean )
	for i in iSort:
		out = ( zips[i], mean[i], stds[i], nRst[i] )
		print '("%s", %3.10f, %3.10f, %d), \\' % out

#####################################################################

# Q1
# max(inspdate) is an aggregating function for the groupby
#
if True:
	q1 = '''
	select zipcode, count(camis) as ncami, sum(score), sum(score*score)
		from 
			(
			select zipcode, camis, score, max(inspdate)
			from data
			group by camis
			)
		group by zipcode
		having ncami > 100
	'''
	# len 208
	# order by count(camis) desc

	cur.execute(q1)
	zgroup = cur.fetchall()

	print " Parsing zipcode results : "
	parseZgroup(zgroup)
	print
	danpause()

		
#####################################################################

if True:
	q2 = '''
	select boroname, count(camis), sum(score), sum(score*score)
	from 
		(
		select boroname, camis, score, max(inspdate)
			from data
				join boro
				on boro.bid = data.boro
				group by camis
		)
		group by boroname
	'''

	cur.execute(q2)
	zgroup = cur.fetchall()

	# data format the same as zipcode, so using the same parser
	print " Parsing borough results : "
	parseZgroup(zgroup)
	print
	danpause()

######################################################################

#"CUISINECODE","CODEDESC"
#	(cuisine, mean score, stderr, number of reports)

if True:
	# returns cuisine, mean, variance, counts
	q4 = '''
	select cuis, sum(score)/count(camis),  ((sum(score*score)/count(camis)) - (sum(score)/count(camis)*sum(score)/count(camis)))/count(camis), count(camis)
		from 
			(
			select codedesc as cuis, camis, score
				from data		
				join cuisine
				on cuisine.cuisinecode = data.cuisinecode
			)
		group by cuis
		having count(cuis) > 100
	order by sum(score)/count(camis) asc
	'''

	cur.execute(q4)
	cgroup = cur.fetchall()

	# rather than installing sqrt extensions for sql, I had numpy do it
	print " Parsing cuisine results : "
	for c in cgroup:
		out = ( c[0], c[1], np.sqrt(c[2]), c[3] )
		print '("%s", %3.10f, %3.10f, %d), \\' % out
	print
	danpause()


#####################################################################

