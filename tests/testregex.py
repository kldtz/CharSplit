#!/usr/bin/env python3

import unittest

from kirke.utils.corenlputils import transform_corp_in_text

class TestRegexUtils(unittest.TestCase):

    def test_transform_corp_in_text1(self):
        "Test transform_corp_in_text()"

        self.assertEqual(transform_corp_in_text("Limited inc has"),
                         "Limited Inc has")
        self.assertEqual(transform_corp_in_text("Limited,inc has"),
                         "Limited Inc has")
        self.assertEqual(transform_corp_in_text("Limited, inc has"),
                         "Limited  Inc has")
        """
        self.assertEqual(transform_corp_in_text("Limited inc"),
                         "Limited Inc")
        self.assertEqual(transform_corp_in_text("Limited,inc"),
                         "Limited Inc")
        self.assertEqual(transform_corp_in_text("Limited, inc"),
                         "Limited  Inc")
        """

        self.assertEqual(transform_corp_in_text("Limited llc has"),
                         "Limited Llc has")        
        self.assertEqual(transform_corp_in_text("Limited,llc has"),
                         "Limited Llc has")
        self.assertEqual(transform_corp_in_text("Limited, llc has"),
                         "Limited  Llc has")
        """
        self.assertEqual(transform_corp_in_text("Limited llc"),
                         "Limited Llc")
        self.assertEqual(transform_corp_in_text("Limited,llc"),
                         "Limited Llc")
        self.assertEqual(transform_corp_in_text("Limited, llc"),
                         "Limited  Llc")
        """

        self.assertEqual(transform_corp_in_text("Limited corp has"),
                         "Limited Corp has")
        self.assertEqual(transform_corp_in_text("Limited,corp has"),
                         "Limited Corp has")
        self.assertEqual(transform_corp_in_text("Limited, corp has"),
                         "Limited  Corp has")
        """
        self.assertEqual(transform_corp_in_text("Limited corp"),
                         "Limited Corp")
        self.assertEqual(transform_corp_in_text("Limited,corp"),
                         "Limited Corp")
        self.assertEqual(transform_corp_in_text("Limited, corp"),
                         "Limited  Corp")
        """

        self.assertEqual(transform_corp_in_text("Limited ltd has"),
                         "Limited Ltd has")
        self.assertEqual(transform_corp_in_text("Limited,ltd has"),
                         "Limited,Ltd has")
        self.assertEqual(transform_corp_in_text("Limited, ltd has"),
                         "Limited, Ltd has")
        """
        self.assertEqual(transform_corp_in_text("Limited ltd"),
                         "Limited Ltd")        
        self.assertEqual(transform_corp_in_text("Limited,ltd"),
                         "Limited,Ltd")
        self.assertEqual(transform_corp_in_text("Limited, ltd"),
                         "Limited, Ltd")
        """

    def test_transform_corp_in_text2(self):
        "Test transform_corp_in_text2()"
        self.assertEqual(transform_corp_in_text("Limited inc has"),
                         "Limited Inc has")        
        self.assertEqual(transform_corp_in_text("Limited,inc has"),
                         "Limited Inc has")
        self.assertEqual(transform_corp_in_text("Limited, inc has"),
                         "Limited  Inc has")
        self.assertEqual(transform_corp_in_text("Limited inc"),
                         "Limited Inc")
        self.assertEqual(transform_corp_in_text("Limited,inc"),
                         "Limited Inc")
        self.assertEqual(transform_corp_in_text("Limited, inc"),
                         "Limited  Inc")

        self.assertEqual(transform_corp_in_text("Limited llc has"),
                         "Limited Llc has")        
        self.assertEqual(transform_corp_in_text("Limited,llc has"),
                         "Limited Llc has")
        self.assertEqual(transform_corp_in_text("Limited, llc has"),
                         "Limited  Llc has")
        self.assertEqual(transform_corp_in_text("Limited llc"),
                         "Limited Llc")
        self.assertEqual(transform_corp_in_text("Limited,llc"),
                         "Limited Llc")
        self.assertEqual(transform_corp_in_text("Limited, llc"),
                         "Limited  Llc")

        self.assertEqual(transform_corp_in_text("Limited corp has"),
                         "Limited Corp has")        
        self.assertEqual(transform_corp_in_text("Limited,corp has"),
                         "Limited Corp has")
        self.assertEqual(transform_corp_in_text("Limited, corp has"),
                         "Limited  Corp has")
        self.assertEqual(transform_corp_in_text("Limited corp"),
                         "Limited Corp")
        self.assertEqual(transform_corp_in_text("Limited,corp"),
                         "Limited Corp")
        self.assertEqual(transform_corp_in_text("Limited, corp"),
                         "Limited  Corp")

        self.assertEqual(transform_corp_in_text("Limited ltd has"),
                         "Limited Ltd has")        
        self.assertEqual(transform_corp_in_text("Limited,ltd has"),
                         "Limited,Ltd has")
        self.assertEqual(transform_corp_in_text("Limited, ltd has"),
                         "Limited, Ltd has")
        self.assertEqual(transform_corp_in_text("Limited ltd"),
                         "Limited Ltd")
        self.assertEqual(transform_corp_in_text("Limited,ltd"),
                         "Limited,Ltd")
        self.assertEqual(transform_corp_in_text("Limited, ltd"),
                         "Limited, Ltd")


    def test_transform_corp_in_text3(self):
        "Test transform_corp_in_text3()"
        self.assertEqual(transform_corp_in_text("Limited inc has"),
                         "Limited Inc has")        
        self.assertEqual(transform_corp_in_text("Limited,inc has"),
                         "Limited Inc has")
        self.assertEqual(transform_corp_in_text("Limited, inc has"),
                         "Limited  Inc has")
        self.assertEqual(transform_corp_in_text("Limited inc"),
                         "Limited Inc")
        self.assertEqual(transform_corp_in_text("Limited,inc"),
                         "Limited Inc")
        self.assertEqual(transform_corp_in_text("Limited, inc"),
                         "Limited  Inc")

        self.assertEqual(transform_corp_in_text("Limited llc has"),
                         "Limited Llc has")        
        self.assertEqual(transform_corp_in_text("Limited,llc has"),
                         "Limited Llc has")
        self.assertEqual(transform_corp_in_text("Limited, llc has"),
                         "Limited  Llc has")
        self.assertEqual(transform_corp_in_text("Limited llc"),
                         "Limited Llc")
        self.assertEqual(transform_corp_in_text("Limited,llc"),
                         "Limited Llc")
        self.assertEqual(transform_corp_in_text("Limited, llc"),
                         "Limited  Llc")

        self.assertEqual(transform_corp_in_text("Limited corp has"),
                         "Limited Corp has")        
        self.assertEqual(transform_corp_in_text("Limited,corp has"),
                         "Limited Corp has")
        self.assertEqual(transform_corp_in_text("Limited, corp has"),
                         "Limited  Corp has")
        self.assertEqual(transform_corp_in_text("Limited corp"),
                         "Limited Corp")
        self.assertEqual(transform_corp_in_text("Limited,corp"),
                         "Limited Corp")
        self.assertEqual(transform_corp_in_text("Limited, corp"),
                         "Limited  Corp")

        self.assertEqual(transform_corp_in_text("Limited ltd has"),
                         "Limited Ltd has")        
        self.assertEqual(transform_corp_in_text("Limited,ltd has"),
                         "Limited,Ltd has")
        self.assertEqual(transform_corp_in_text("Limited, ltd has"),
                         "Limited, Ltd has")
        self.assertEqual(transform_corp_in_text("Limited ltd"),
                         "Limited Ltd")
        self.assertEqual(transform_corp_in_text("Limited,ltd"),
                         "Limited,Ltd")
        self.assertEqual(transform_corp_in_text("Limited, ltd"),
                         "Limited, Ltd")    

    def test_transform_corp_in_text4(self):
        self.assertEqual(transform_corp_in_text("eBrevia inc is a start-up company."),
                                                "eBrevia Inc is a start-up company.")
        self.assertEqual(transform_corp_in_text("eBrevia ltd is a start-up company."),
            "eBrevia Ltd is a start-up company.")
        self.assertEqual(transform_corp_in_text("eBrevia llc is a start-up company."),
                                                "eBrevia Llc is a start-up company.")
        self.assertEqual(transform_corp_in_text("eBrevia corp is a start-up company."),
                                                "eBrevia Corp is a start-up company.")
        self.assertEqual(transform_corp_in_text("eBrevia, inc is a start-up company."),
                                                "eBrevia  Inc is a start-up company.")
        self.assertEqual(transform_corp_in_text("eBrevia, ltd is a start-up company."),
                                                "eBrevia, Ltd is a start-up company.")
        self.assertEqual(transform_corp_in_text("eBrevia, llc is a start-up company."),
                                                "eBrevia  Llc is a start-up company.")
        self.assertEqual(transform_corp_in_text("eBrevia, corp is a start-up company."),
                                                "eBrevia  Corp is a start-up company.")
        self.assertEqual(transform_corp_in_text("eBrevia, inc. works with Venture, LTD."),
                                                "eBrevia  Inc. works with Venture, Ltd.")
        self.assertEqual(transform_corp_in_text("eBrevia, iNc. works with Baker, llC."),
                                                "eBrevia  Inc. works with Baker  Llc."),
        self.assertEqual(transform_corp_in_text("EBREVIA, INC. PROFIT INCREASES."),
                                                "EBREVIA  Inc. PROFIT INCREASES.")
        self.assertEqual(transform_corp_in_text("EBREVIA, LTD. IS PRIVATE LIMITED."),
                                                "EBREVIA, Ltd. IS PRIVATE LIMITED.")
        self.assertEqual(transform_corp_in_text("EBREVIA LLC. DEVELOPS AI."),
                                                "EBREVIA Llc. DEVELOPS AI.")
        self.assertEqual(transform_corp_in_text("EBREVIA, CORP. INCOPORATES."),
                                                "EBREVIA  Corp. INCOPORATES.")
        self.assertEqual(transform_corp_in_text("INCORPORATION"),
                                                "INCORPORATION")
        self.assertEqual(transform_corp_in_text("Coporation"),
                                                "Coporation")
        self.assertEqual(transform_corp_in_text("THISISINCORP.ORATION."),
                                                "THISISINCORP.ORATION.")
        self.assertEqual(transform_corp_in_text("THIS IS eBrevia, INC."), 
                                                "THIS IS eBrevia  Inc.")
        self.assertEqual(transform_corp_in_text("THIS IS eBrevia,INC."), 
                                                "THIS IS eBrevia Inc.")
        self.assertEqual(transform_corp_in_text("THIS IS eBrevia,LTD."), 
                                                "THIS IS eBrevia,Ltd.")
        self.assertEqual(transform_corp_in_text("THIS IS eBrevia,ABCDELTD."),
                                                "THIS IS eBrevia,ABCDELTD.")
        self.assertEqual(transform_corp_in_text("INC."),
                                                "Inc.")
        self.assertEqual(transform_corp_in_text("lTD")
                                                ,"Ltd")
        self.assertEqual(transform_corp_in_text(",lTD "),
                                                ",Ltd ")
        self.assertEqual(transform_corp_in_text(",INC."),
                                                " Inc.")
        self.assertEqual(transform_corp_in_text(", INC."),
                                                "  Inc.")
        self.assertEqual(transform_corp_in_text("LTD.")
                                                ,"Ltd.")
        self.assertEqual(transform_corp_in_text(",Ltd."),
                                                ",Ltd.")
        self.assertEqual(transform_corp_in_text(", LTd."),
                                                ", Ltd.")
        self.assertEqual(transform_corp_in_text("(THIS IS eBrevia INC)"),
                                                "(THIS IS eBrevia Inc)")

if __name__ == "__main__":
    unittest.main()

