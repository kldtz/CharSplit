import re
from kirke.docstruct import parav2finder as lxparav2
from kirke.docstruct import lxlineinfo
from kirke.utils import mathutils

TOC_PREFIX_PAT = re.compile(r'^\s*(table\s*of\s*contents?)\s*$', re.IGNORECASE)
EXHIBIT_PAT = re.compile(r'^(exhibit|exmllit|appendix) ', re.IGNORECASE)

TOC_IN_FIRST_N_PAGES = 8

def mark_tocline(page_lineinfos_list):
    # assume table of content is in the first 5 pages
    result = []
    for page in page_lineinfos_list[:TOC_IN_FIRST_N_PAGES]:
        page_toc_linfos = []
        for lineinfo in page:
            if TOC_PREFIX_PAT.match(lineinfo.text):
                lineinfo.category = 'toc-header'
                page_toc_linfos.append(lineinfo)
            elif '.....' in lineinfo.text:
                lineinfo.category = 'tocline'
                page_toc_linfos.append(lineinfo)

        if page_toc_linfos:
            for lineinfo in page:            
                # print("checking [{}]".format(lineinfo.text))
                mat = EXHIBIT_PAT.match(lineinfo.text)
                if mat:
                    lineinfo.category = 'tocline'                    
                    page_toc_linfos.append(lineinfo)
            result.extend(sorted(page_toc_linfos))  # make sure that they are ordered
    return result


def get_page_top_start(lineinfo_list, pheader_list):
    for lineinfo in lineinfo_list:
        if lineinfo not in pheader_list:
            return lineinfo.start
    return -1


def extract_toc_parav2s(page_lineinfos_list, doc_text, pheader_linfo_list, pfooter_linfo_list, pagenum_linfo_list):
    toc_lineinfos = set(mark_tocline(page_lineinfos_list))

    parav2_results = []
    lineinfo_results = []    
    num_toc = 0
    # take only the first 8 pages
    for page_num, page in enumerate(page_lineinfos_list[:TOC_IN_FIRST_N_PAGES], 1):

        # print("===== page #{} ========================".format(page_num))
        norm_lineinfos = []
        toc_linfos = []
        num_very_short_sent = 0
        num_not_english = 0 
        for linfo in page:
            if linfo in toc_lineinfos:
                # print("found toc: {}".format(linfo.text))
                num_toc += 1
                toc_linfos.append(linfo)
            elif not (linfo in pheader_linfo_list or linfo in pfooter_linfo_list or linfo in pagenum_linfo_list):
                # print("in norm....{}".format(linfo.text[:40]))
                norm_lineinfos.append(linfo)
                if len(linfo.text) < 20:
                    num_very_short_sent += 1
                if not linfo.is_english:
                    num_not_english += 1
            #else:
            #    print("in skip list")

        toc_linfos_set = set(toc_linfos)   # a temporary set to check for add to toc_linfos
        # some lines are not toc marked, but should be, such as "FORM OF CONSENT AND AGREEMENT" after number
        if len(toc_linfos) >= 5 and ((num_very_short_sent + len(toc_linfos)) / len(page) > 0.8 or
                                     (num_not_english + len(toc_linfos)) / len(page) > 0.8):
            for linfo in norm_lineinfos:
                linfo.category = 'tocline'
                if linfo not in toc_linfos_set:
                    toc_linfos.append(linfo)

        if toc_linfos:
            start, end = lxparav2.calc_lineinfos_start_end(toc_linfos)

            full_toc_linfos = []
            page_top_to_toc_end_linfos = []
            page_top_start = get_page_top_start(page, pheader_linfo_list)
            for linfo in page:
                if mathutils.start_end_overlap((start, end), (linfo.start, linfo.end)):
                    full_toc_linfos.append(linfo)
                if mathutils.start_end_overlap((page_top_start, end), (linfo.start, linfo.end)):
                    page_top_to_toc_end_linfos.append(linfo)

            if lxlineinfo.is_lineinfos_word_overlap(full_toc_linfos, page_top_to_toc_end_linfos, 0.8):
                para2 = lxparav2.init_parav2('toc', page_top_to_toc_end_linfos, doc_text)
                for linfo in page_top_to_toc_end_linfos:
                    linfo.category = 'toc'
                para2.lineinfo_list = page_top_to_toc_end_linfos
            else:
                para2 = lxparav2.init_parav2('toc', full_toc_linfos, doc_text)
                for linfo in full_toc_linfos:
                    linfo.category = 'toc'
                para2.lineinfo_list = full_toc_linfos
            parav2_results.append(para2)
            lineinfo_results.extend(para2.lineinfo_list)
    
    return parav2_results, lineinfo_results

    
