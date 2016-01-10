## for the final problem in the SQL miniproject

import pandas as pd
import sqlite3
from dan import danpause
import numpy as np

db = sqlite3.connect('nycql.db')
cur = db.cursor()

# total violations : 531933



# total violations by code:
#	02A 498
#	02B 24523
#	02C 550
#	...
#	22C 4401
#	22E 61
#	99B 2257
nViol_byCode = '''
select violcode as vcode, count(*) as vcount, VIOLATIONDESC as descrip
	from data		
	join
		(select * 
			from violations 
			where enddate > "2014-01-01 00:00:00"
		) as viol
	on data.violcode = viol.violationcode
group by violcode
'''
#	len = 93



# total violations by cuisine type:
#	Afghan 307
#	African 1877
#	American  128372
#	...
#	Turkish 1557
#	Vegetarian 1983
#	Vietnamese/Cambodian/Malaysia 1856
nViol_byCuisine = '''
select cuis, count(*) as ccount
	from 
	(
	select codedesc as cuis, violcode as vcode
		from data		
		join cuisine
		on cuisine.cuisinecode = data.cuisinecode
	)
group by cuis
'''
#	len = 84




nViol_byCodeCuisine = '''
select cuis, vcode, count(*) as condcount
	from 
		(
		select codedesc as cuis, violcode as vcode
			from data		
			join cuisine
			on cuisine.cuisinecode = data.cuisinecode
		)
group by cuis, vcode
having count(vcode) > 100
'''
#	len = 773, has [ cuis, violcode, count]
#		same result with:
#	 		having count(cuis) > 100
#			having count(violcode) > 100




#	select 
#	from nViol_byCuisine
#	join nViol_byCodeCuisine
#	produces conditional probabilities of violation given cuisine
kilo = '''
	select A.cuis, A.vcode, A.condcount * 1.0/B.ccount as freq
	FROM
		(
		select cuis, vcode, count(*) as condcount
		from 
			(
			select codedesc as cuis, violcode as vcode
				from data		
				join cuisine
				on cuisine.cuisinecode = data.cuisinecode
			)
		group by cuis, vcode
		having count(vcode) > 100
		) as A
	JOIN
		(
		select cuis, count(*) as ccount
		from 
			(
			select codedesc as cuis, violcode as vcode
				from data		
				join cuisine
				on cuisine.cuisinecode = data.cuisinecode
			)
		group by cuis
		) as B

	ON A.cuis = B.cuis
'''


# given conditional described in Kilo above, check vs probability of each code
#	select 
#	from nViol_byCuisine
#	join nViol_byCodeCuisine
#	join nViol_byCode
#	produces conditional probabilities of violation given cuisine

mega = '''
	select A.cuis, descrip, A.condcount * 1.0/B.ccount * 531933.0/C.vcount as freq, A.condcount
	FROM
		(
		select cuis, vcode, count(*) as condcount
		from 
			(
			select codedesc as cuis, violcode as vcode
				from data		
				join cuisine
				on cuisine.cuisinecode = data.cuisinecode
			)
		group by cuis, vcode
		having count(vcode) > 100
		) as A
	JOIN
		(
		select cuis, count(*) as ccount
		from 
			(
			select codedesc as cuis, violcode as vcode
				from data		
				join cuisine
				on cuisine.cuisinecode = data.cuisinecode
			)
		group by cuis
		) as B
		ON A.cuis = B.cuis

	JOIN
		(
		select violcode as vcode, count(*) as vcount, VIOLATIONDESC as descrip
			from data		
			join
				(select * 
					from violations 
					where enddate > "2014-01-01 00:00:00"
				) as viol
			on data.violcode = viol.violationcode
		group by violcode
		) as C
		on A.vcode = C.vcode

	order by freq desc
	limit 25
'''

	cur.execute(mega)
	gmega = cur.fetchall()
	#531933 is total violations

	for v in gmega:
		out = ( v[0], v[1], v[2], v[3] ) 
		print '( ("%s", "%s"), %3.10f, %d), \\' % out

