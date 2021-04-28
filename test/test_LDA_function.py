import sys
sys.path.append('../src/')
import LDA_function
import unittest
import numpy as np



class TestLDA(unittest.TestCase):

    def test_lda_function(self):

        # Sample data for test
        d1 = "Java is a language for programming that develops a software for several platforms. A compiled code or bytecode on Java application can run on most of the operating systems including Linux, Mac operating system, and Linux. Most of the syntax of Java is derived from the C++ and C languages."
        d2 = "Python supports multiple programming paradigms and comes up with a large standard library, paradigms included are object-oriented, imperative, functional and procedural."
        d3 = "Go is typed statically compiled language. It was created by Robert Griesemer, Ken Thompson, and Rob Pike in 2009. This language offers garbage collection, concurrency of CSP-style, memory safety, and structural typing."
        d4 = "A young girl when she first visited magical Underland, Alice Kingsleigh (Mia Wasikowska) is now a teenager with no memory of the place -- except in her dreams."
        d5 = "Her life takes a turn for the unexpected when, at a garden party for her fiance and herself, she spots a certain white rabbit and tumbles down a hole after him. Reunited with her friends the Mad Hatter (Johnny Depp), the Cheshire Cat and others, Alice learns it is her destiny to end the Red Queen's (Helena Bonham Carter) reign of terror."
        
        # Using tf function to prepare the input data for my_lda_function
        tf_df, id2word = LDA_function.tf([d1, d2, d3, d4, d5])

        lil = []
        for row in tf_df.values:
            lil_sub = []
            for idx, item in enumerate(row):
                if item:
                    lil_sub.append((idx, item))
            lil.append(lil_sub)

        shown, gamma_by_chunks = LDA_function.my_lda_func(corpus=lil, num_topics=2, id2word=id2word, topics_only=False, num_words=10, verbose=False, passes=10)        
        
        self.assertEqual(len(shown), 2)
        shown = sorted(shown, key=lambda x: x[1])
        
        self.assertEqual(shown[0][1], 
  '0.023*"language" + 0.014*"alice" + 0.014*"memory" + 0.014*"compiled" + 0.014*"concurrency" + 0.014*"go" + 0.014*"safety" + 0.014*"griesemer" + 0.014*"collection" + 0.014*"csp"')
        
        self.assertEqual(shown[1][1], 
  '0.032*"java" + 0.023*"operating" + 0.023*"linux" + 0.023*"paradigms" + 0.023*"c" + 0.023*"programming" + 0.014*"language" + 0.014*"compiled" + 0.014*"systems" + 0.014*"mac"')
        
        self.assertEqual(len(gamma_by_chunks), 5)
    
if __name__ == '__main__':
    unittest.main()
