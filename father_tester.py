#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 17:57:47 2024

@author: vanessa
"""

#change / or \ line 72 approx (depending on system)

import avn.similarity as similarity
import pandas as pd
import os
import csv
from itertools import combinations
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


class Bird:
    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.birdID = os.path.basename(folder_path)
        self.csv_path = ""
        
        #debugging
        print(f"Bird ID is: {self.birdID}")
        
        self.embeddings = np.array([[]])

    
    def getSeg_csv(self):
        '''Get segmentation csv file for this bird
        :param song_folder: Path to dir with .wav files + segmentation csv 
        :return: Tuple/list (segmentation file path, bird ID)
        '''
        seg_path = os.path.join(self.folder_path, 'Segmentations')
        seg_file = os.path.join(seg_path, f"{self.birdID}.csv")     
        
        #debugging 
       # print(f"Bird name : {self.birdID}")
        #print(f"Segmentations Path: {seg_path}")  # Debugging line
       # print(f"CSV File Path: {seg_file}")  # Debugging line
        
        
        return seg_file
    
    
    def get_embedding(self):
        ''' 
        Calculate and return embeddings 
        '''
        if self.embeddings.size > 0:   # if alr calculated, return it
            return self.embeddings
            
        seg_path = self.getSeg_csv()    # get segmentation csv path
        folder_path = self.folder_path
        
        #debuggin 
       # print("Got to get_embeddings\n")
        print(f"CSV Path: {seg_path}")  # Debugging line
        print(f"Song folder Path: {self.folder_path}")  # Debugging line
        print(f"BirdID: {self.birdID}")
        
        segmentations = pd.read_csv(seg_path)
        segmentations.head()
        

        output_dir = os.path.join(self.folder_path, 'Embeddings') #put embeddings in a folder under that bird
        os.makedirs(output_dir, exist_ok=True)  # Create the folder if it doesn't exist
        folder_path = folder_path + '/'
        similarity.prep_spects(Bird_ID=self.birdID, segmentations=segmentations,        # generate spectrograms ("594 sylable spectrograms saved ...)
                               song_folder_path=folder_path, out_dir=output_dir)
        
        #debug
        print("done prep")
    
        embeddings_first = np.array([[]])
        
        model = similarity.load_model()  # Load the pre-trained model
        embeddings = similarity.calc_embeddings(Bird_ID=self.birdID, 
                                                     spectrograms_dir=output_dir, 
                                                     model=model)
        self.embeddings = embeddings 
        
        #debug
        print("Done embeddings")
        print(self.embeddings)
        return self.embeddings
        
        
class SimilarityCalculator: 
    def __init__(self, bird1: Bird, bird2: Bird):
        self.bird1 = bird1
        self.bird2 = bird2
    
    
    def calc_score(self):
        '''
        Returns: similarity score between 2 birds
        '''
        embedding_1 = self.bird1.get_embedding()
        embedding_2 = self.bird2.get_embedding()  
       
        emd = similarity.calc_emd(embedding_1, embedding_2)
        print(f"emd is: {emd}")
        return emd
    


class ParseDirectory: 
    def __init__(self, dir_path):
        self.dir_path = dir_path
        #self.output_path = output_path 
        
        
    def process_directory(self):
        '''
        Generator yield subdirectory paths (bird folders in the parent directory)'
        '''
        for entry in os.listdir(self.dir_path):
            subdir_path = os.path.join(self.dir_path, entry)
            if os.path.isdir(subdir_path):
                yield subdir_path
    
    def process_all(self, output_dir):
        '''
        Processes all birds combinatinos within directory and computes emd, saves result to a csv file
        '''
        output_csv_path = os.path.join(output_dir, f"{os.path.basename(self.dir_path)}.csv")
        
        subdirs = list(self.process_directory()) # makes a list of all subdirectory paths 
        embeddings_cache = {}   # dictionary to store embeddings for each bird
        
        for subdir in subdirs:                      # pre cache every bird
            bird = Bird(subdir)
            bird.get_embedding()                    # calculate the embedding
            embeddings_cache[bird.birdID] = bird    #cache the bird
        
        with open(output_csv_path, "w", newline='') as outputcsv:   # make csv file
            writer = csv.writer(outputcsv)
            writer.writerow(['Bird 1', 'Bird 2', 'SIMILARITY SCORE']) # create header
        
              
            for (bird_id1, bird1), (bird_id2, bird2) in combinations(embeddings_cache.items(), 2):  # iterate each bird combo in cache
                similarity_calculator = SimilarityCalculator(bird1, bird2)                          # create score calculator object
                emd_score = similarity_calculator.calc_score()                                      # get score
                
                writer.writerow([bird1.birdID, bird2.birdID, emd_score])                            # write birdIDs + score into csv
                
        print(f"Similarity Score CSV created at {output_csv_path}")
        
        
    def compare_multiple_to_one(self, reference_folder, output_dir): # self is the compare_folder
         '''
         Compares all birds in the current directory to the reference folder (tutor or pre)
         '''
         output_csv_path = os.path.join(output_dir, f"{os.path.basename(self.dir_path)}.csv")
         reference_bird = Bird(reference_folder)           # create reference bird object 
         subdirs = list(self.process_directory())          # makes a list of all subdirectory paths in current dir
         
         with open(output_csv_path, "w", newline='') as outputcsv:
             writer = csv.writer(outputcsv)
             writer.writerow(['Reference Bird ID','Bird ID', 'Similarity Score'])
            
             for i in subdirs:
                 compared_bird = Bird(i)                    
                 calculator = SimilarityCalculator(reference_bird, compared_bird)   
                 emd = calculator.calc_score()                    # get score
                 writer.writerow([reference_bird.birdID, compared_bird.birdID, emd])  # write to csv
                
         print (f"Similarity Score to Reference bird CSV created at {output_csv_path}") # print confirmation
        
                

class MasterComparer:
    def __init__(self, father_dir, pupil_dir, output_path):
        self.pupil_dir = pupil_dir
        self.father_dir = father_dir
        self.pupil_dictionary = {}      
        self.output_path = output_path 


    def get_pupil_name(self, pupil_path):
        '''
        Creates pupil object if it doesnt exist already, returns false. Returns true if already exists
        '''
        pupil_name = os.path.basename(pupil_path)   # gets pupils name from its folder name 
        if pupil_name not in self.pupil_dictionary:      # if pupil is not in dictionary, create bird object and return false
           self.pupil_dictionary[pupil_name] = Bird(pupil_path)

      
                
    def tutordir_vs_pupildir(self):
         '''
         Compares all fathers to each child. Each input is a master directory (all fathers vs all pulils)
         '''
         output_csv_path = os.path.join(self.output_path, "tutor_comparisons.csv")    # creates output csv 
         father_subdirs = gen_process_directory(self.father_dir)          # get father subdirectories  
        # pupil_subdirs = gen_process_directory(self.pupil_dir)                    # get pupil subdirectories
         
         
         with open(output_csv_path, "w", newline='') as outputcsv:
             writer = csv.writer(outputcsv)
             writer.writerow(['Tutor Bird ID','Pupil Bird ID', 'Similarity Score'])
         
             for father in father_subdirs:
                 father_bird = Bird(father)      # create father Bird object
                 pupil_subdirs = gen_process_directory(self.pupil_dir) # get list of pupil sub directories
                 
                 for pupil in pupil_subdirs:
                     pupil_timepoints = gen_process_directory(pupil)     # get songs in general BirdID folder
                     
                     for timepoint in pupil_timepoints:
                         timepoint_name = os.path.basename(timepoint) # get name of specific timepoint
                         
                         
                         self.get_pupil_name(timepoint)          # create Bird object if it doesnt exist yet
                         calculator = SimilarityCalculator(father_bird, self.pupil_dictionary[os.path.basename(timepoint)])   # calculate emd
                         emd = calculator.calc_score()                    # get score
                         writer.writerow([os.path.basename(father), os.path.basename(timepoint), emd])  # write to csv
                         
                         print(f"Done {os.path.basename(timepoint)} vs {os.path.basename(father)}")
                            
                 
                
# general directory processor      
   
def gen_process_directory(directory):
     '''
     Generator yield subdirectory paths (bird folders in the parent directory)'
     '''
     for entry in os.listdir(directory):
         subdir_path = os.path.join(directory, entry)
         if os.path.isdir(subdir_path):
             yield subdir_path          
# System and user input checks 

  
    
if __name__ == "__main__":
    
    '''
    
    choice = input("Enter 1 to compare all birds to each other in a directory \n Enter 2 to compare 2 birds only \n Enter 3 to compare a directory of birds to a single reference bird\nType exit to stop\n: ")
    
    if choice == "1": # Option 1: Compare all 
        cur_dir = input("Enter path of directory to compare: ")
        output_dir = input("Enter desired OUTPUT directory path: ")
        parser1 = ParseDirectory(cur_dir)           
        parser1.process_all(output_dir)     # actual computation
        print(f"Completed. Check {output_dir} ")
        
    elif choice == "2": # Option 2: Compare just 2 birds (time points)
        first_dir = input("Enter path of first directory: ")
        
        print(f"Path entered: {first_dir}")
        
        second_dir = input("Enter path of second directory: ")
        output_dir = input("Enter desired OUTPUT directory path if needed\nElse, enter no: ")
        bird1 = Bird(first_dir)     # instantiate bird1
        bird2 = Bird(second_dir)    # instantiate bird2
        sim_calc = SimilarityCalculator(bird1, bird2)   # make calculator 
        score = sim_calc.calc_score()
        
        if output_dir!= "no":
            os.makedirs(output_dir, exist_ok=True)
            # making a csv if needed
            output_csv_path = os.path.join(output_dir, f"{os.path.basename(first_dir)}vs{os.path.basename(second_dir)}.csv")
            
            with open(output_csv_path, "w", newline='') as outputcsv:
                writer = csv.writer(outputcsv)
                writer.writerow(['Bird 1','Bird 2', 'Similarity Score'])
                writer.writerow([bird1.birdID, bird2.birdID, score])
        
        print(f"Completed\nSimilarity Score is: {score}\nCheck {output_dir} if csv wanted")
        
        
        
    elif choice == "3": # Option 3: Compare many timepoints to a single reference point
        reference_folder = input("Enter path of reference bird: ")
        compare_folder = input("Enter path of directory to be compared: ")
        output_dir = input("Enter desired OUTPUT directory path: ")

        parser3 = ParseDirectory(compare_folder)
        parser3.compare_multiple_to_one(reference_folder, output_dir)       # computing

        print(f"Completed. Check {output_dir} ")
        
        
        '''
    # Use this to compare a directory of fathers to a directory of pupils    
    father_dir = "/Users/vanessa/Documents/fathers"
    pupil_dir = "/Users/vanessa/Documents/Extra_Songs"
    output_path = "/Users/vanessa/Documents/SAKATA_LAB_OUTPUTS"
    MasterComparereObj = MasterComparer(father_dir, pupil_dir, output_path)
    MasterComparereObj.tutordir_vs_pupildir()
        
        
        
        
        
        
    