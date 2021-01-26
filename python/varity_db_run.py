import sys
import alm_humandb

#initiate varity 
db_obj = alm_humandb.alm_humandb(sys.argv)
db_obj.humandb_action(db_obj.runtime)
       


