#!/usr/bin/env python3

import unittest

from kirke.docstruct import secheadutils

def parse_sechead(line, prev_line='', prev_line_idx=-1):
    sechead_type, prefix, sechead, split_idx = secheadutils.extract_sechead_v4(line,
                                                                               prev_line=prev_line,
                                                                               prev_line_idx=prev_line_idx)
    return prefix, sechead
    
class TestSecHeadUtils(unittest.TestCase):

    def test_one_line(self):
        "Test transform_corp_in_text()"

        self.assertEqual(parse_sechead('Appendix A'),
                         ('Appendix A', ''))
        self.assertEqual(parse_sechead('Appendix A:'),
                         ('Appendix A', ''))

        self.assertEquals(parse_sechead('3.2.2           Percentage Adjustment. The Monthly Base Rent shall...'),
                          ('3.2.2', 'Percentage Adjustment. '))
        self.assertEquals(parse_sechead('G&I V Midwest Residential LLC (2)'),
                          ('', ''))
        self.assertEquals(parse_sechead('H Exhibit'),
                          ('Exhibit H', ''))
        self.assertEquals(parse_sechead('7.7. Complete Agreement.'),
                          ('7.7.', 'Complete Agreement.'))
        self.assertEquals(parse_sechead('Signature: /s/Jim Perkins'),
                          ('', ''))
        self.assertEquals(parse_sechead('C.3 STRATEGIC SALES & MARKETING AGREEMENT'),
                          ('C.3', 'STRATEGIC SALES & MARKETING AGREEMENT'))
        self.assertEquals(parse_sechead('Recitals'),
                          ('Recitals', ''))
        self.assertEquals(parse_sechead('Exhibit H—Tenant Estoppel Certificate.'),
                          ('Exhibit H', 'Tenant Estoppel Certificate.'))
        self.assertEquals(parse_sechead('ARTICLE II. CERTAIN COVENANTS'),
                          ('ARTICLE II.', 'CERTAIN COVENANTS'))
        self.assertEquals(parse_sechead('Exhibit B-2'),
                          ('Exhibit B-2', ''))
        self.assertEquals(parse_sechead('A Mexican Corporation'),
                          ('', ''))
        self.assertEquals(parse_sechead('A G R E E M E N T'),
                          ('', ''))        
        self.assertEquals(parse_sechead('AGREEMENT'),
                          ('', 'AGREEMENT'))
        self.assertEquals(parse_sechead('944-6464 Mobile'),
                          ('', ''))
        self.assertEquals(parse_sechead('(e)        Regulation S. The Securities will be offered and sold'),
                          ('(e)', 'Regulation S. '))

        self.assertEquals(parse_sechead('(d)   Severability.  In the event that any one or more of'),
                          ('(d)', 'Severability.  '))

        self.assertEquals(parse_sechead('8.1 Indemnification by Tenant. Subject to the...'),
                          ('8.1', 'Indemnification by Tenant. '))
        
                          

    def test_transform_corp_in_text2(self):
        "Test transform_corp_in_text2()"
        self.assertEquals(parse_sechead('Commission Schedule', prev_line='Appendix A:'),
                          ('Appendix A', 'Commission Schedule'))
        self.assertEquals(parse_sechead('9', prev_line='16. Pari Passu Notes. xxx', prev_line_idx=22),
                          ('', ''))
        self.assertEquals(parse_sechead('Services', prev_line='Article II'),
                          ('Article II', 'Services'))
        self.assertEquals(parse_sechead('Assignment', prev_line='Article V'),
                          ('Article V', 'Assignment'))
        self.assertEquals(parse_sechead('Exhibit A', prev_line='5'),
                          ('Exhibit A', ''))
        self.assertEquals(parse_sechead('Exhibit A', prev_line='page 5 of 20'),
                          ('Exhibit A', ''))
        self.assertEquals(parse_sechead('A.', prev_line='Background'),
                          ('', 'Background A.'))
        self.assertEquals(parse_sechead('In this Agreement', prev_line='Definitions'),
                          ('', 'Definitions In this Agreement'))
        self.assertEquals(parse_sechead('Engagement.  Subject to the Terms and Conditions of this...',
                                       prev_line='1.'),
                          ('1.', 'Engagement.  '))
        self.assertEquals(parse_sechead('Exhibit A', prev_line='EXHIBITS'),
                          ('Exhibit A', ''))
        self.assertEquals(parse_sechead('Schedule 2.13', prev_line='SCHEDULES'),
                          ('Schedule 2.13', ''))
        self.assertEquals(parse_sechead('Exhibit B-2', prev_line='Exhibit B-1'),
                          ('Exhibit B-2', ''))
        self.assertEquals(parse_sechead('Exhibit A – Notice of Conversion', prev_line='Exhibit'),
                          ('Exhibit A', 'Notice of Conversion'))

        self.assertEquals(parse_sechead('Operating     Requirements / Performance', prev_line='5.'),
                          ('5.', 'Operating     Requirements / Performance'))


    def test_is_line_sechead_prefix(self):
        "Test is_line_sechead_prefix"
        self.assertTrue(secheadutils.is_line_sechead_prefix('EXHIBITC'))


if __name__ == "__main__":
    unittest.main()

