from astropy import coordinates as coord

#change coordinates of Sculptor from galactic to fk5 
sculptor = coord.SkyCoord(l=287.5, b=-83.25, frame='galactic', unit='deg')
sculptor_fk5 = sculptor.transform_to('fk5')
print(sculptor_fk5)