import requests
from bs4 import BeautifulSoup

import pdb

def nasa_nssdc_scraper(obj_name, discipline='Any Discipline', launch_date=None):
    '''Function to scrape all textual data about a given object from the
       NASA National Space Science Data Center Catalog (NSSDC)

       Parameter:
           obj_name (str): space object name
           discipline (str): mission of the space object
           launch_date (str): date of object launch
       Return:
           str with all textual information on given object in catalog
    '''
    # Post query to nssdc spacecraft search page
    base_nssdc_url = "https://nssdc.gsfc.nasa.gov"
    nssdc_query_url = base_nssdc_url + "/nmc/spacecraft/query"
    #with requests.Session() as rs:
    if launch_date:
        package = {"name" : obj_name,
                   "query_discipline" : discipline,
                   "launch" : launch_date,
                   "submit" : "submit"
                   }
    else:
        package = {"name" : obj_name,
                   "query_discipline" : discipline,
                   "submit" : "submit"
                  }

    rp = requests.post(nssdc_query_url, data=package)
    if rp.status_code != requests.codes.ok:
        print("\nERROR: response not recieved from NASA NSSDC query engine")
        print("\nCheck: {}".format(nssdc_query_url))
        raise Exception

    # Sort out html mumbo jumbo
    soup = BeautifulSoup(rp.content, 'html.parser')

    # parse number of objects returned by query
    num_obj_return_html = soup.find_all('p')[0].get_text()
    num_obj_return = int(num_obj_return_html.split()[2])

    # Return error message if no objects returned by query
    if num_obj_return == 0:
        print("\nERROR: Could find no relevant information on object ")
        print("{}".format(obj_name))
        print("\nTry search again")
        raise Exception

    # grab info for all spacecraft returned by query
    obj_urls_html = soup.find_all('a', href=True)[:num_obj_return+1]
    obj_info_html = soup.find_all('td')

    # Prepare lists for all info we are going to extract for each object:
    #        1. object names
    #        2. object nssdca ids
    #        3. object launch dates
    obj_names = []
    obj_nssdca_ids = []
    obj_launch_dates = []

    # Prepare a comprehensive object information dictionary to contain all
    # pertinent information for each object
    obj_info_dict = {}

    # iterator to keep track of which object we are gathering info for
    obj_iter = 0

    for idx, obj_info in enumerate(obj_info_html):


        if idx % 3 == 0:
            obj_names.append(obj_info.get_text())
        elif idx % 3 == 1:
            obj_nssdca_ids.append(obj_info.get_text())
        else:
            obj_launch_dates.append(obj_info.get_text())
            # Prepare new entry for dictionary. We have all info we
            # need for a specific object
            spec_obj_name = obj_names[obj_iter]
            spec_obj_url = obj_urls_html[obj_iter+1]['href']
            spec_obj_nssdca_id = obj_nssdca_ids[obj_iter]
            spec_obj_launch_date = obj_launch_dates[obj_iter]
            obj_info_dict[spec_obj_name] = [spec_obj_url,
                                            spec_obj_nssdca_id,
                                            spec_obj_launch_date]
            obj_iter += 1

    # List out objects captured by search and ask user what objects they want
    # to include in the corpus
    test_name = obj_names[0]
    obj_info = obj_info_dict[test_name]
    obj_info_url = base_nssdc_url + obj_info[0]
    grab_object_nssdc_info_from_url(obj_info_url)

#    if len(obj_names) == 1:
#        print("\nInclude object name {} in corpus?".format(obj_names[0]))
#        confirmation = input("\n(Y/n)?")
#        if confirmation.lower() == "y":
#            response = save_object_in_corpus(obj_info_dict)
#
#    print("\n")
#    for obj_names, _ in obj_info_dict.items():





def grab_object_nssdc_info_from_url(url):
    '''Grab text from nssdc object page to incorporate into corpus'''
    rs = requests.get(url)
    if rs.status_code != requests.codes.ok:
        print("\nERROR: no response from {}".format(url))
        raise Exception
    soup = BeautifulSoup(rs.content, 'html.parser')
    print(soup.prettify())
    stuff = soup.find_all('p')
    object_summary = soup.find_all('p')[0]
    for stufferino in stuff:
        print(stufferino.get_text())


def astriagraph_scraper(obj_name, data_source='All',  
        nat_of_origin='All', orbit_regime='All'):
    '''Scrapes information of space object from University of 
       Texas, Austin's AstriaGraph.

       http://astria.tacc.utexas.edu/AstriaGraph/

       Parameters:
           obj_name (str) -- name of object
           data_source (str) -- source for information on object
                            possible data sources:
                                'Astria OD/LeoLabs data'
                                'Astria OD/Starbrook data'
                                'JSC Vimpel'
                                'LeoLabs'
                                'Planet'
                                'SeeSat-L'
                                'USSTRATCOM'
            nat_of_origin (str) -- country of origin for the object
            orbit_regime (str) -- orbit regime for object
                            possible orbit regimes:
                                'Low Earth orbit (LEO)'
                                'Medium Earth orbit (MEO)'
                                'Geo-synchronous/stationary orbit (GSO/GEO)'
                                'High Earth Orbit (HEO)'
    '''
    astriagraph_url = "http://astria.tacc.utexas.edu/AstriaGraph/"
    #with requests.Session() as rs:
    package = {"SearchBox" : obj_name,
               "DataSrcSelect" : data_source,
               "OriginSelect" : nat_of_origin,
               "RegimeSelect" : orbit_regime
               }

    rp = requests.post(astriagraph_url, data=package)
    if rp.status_code != requests.codes.ok:
        print("\nERROR: response not recieved from NASA NSSDC query engine")
        print("\nCheck: {}".format(nssdc_query_url))
        raise Exception

    # Sort out html mumbo jumbo
    soup = BeautifulSoup(rp.content, 'html.parser')
    print(soup.prettify())

def google_search_scraper(search_term, output='xml_no_dtd', 
        cx='placeholder', *args, **kwargs):
    '''Performs a query with google and returns results. Further 
       scraping operations can be performed if desired

       Parameters:
           args (str): additional terms that will be used in query
           kwargs (str): google api parameters
    '''
    google_query_url = "http://www.google.com/search?"
    client = 'google-csbe'
    query_str = ''
    for arg in args: query_str + str(arg)


def save_object_info_to_corpus(object_info, *args, **kwargs):
    '''Save scraped object info into text corpus'''
    # TODO: define a dictionary format for textual corpus data
    pass



if __name__ == '__main__':
    #nasa_nssdc_scraper('Galaxy')
    astriagraph_scraper('Galaxy')

