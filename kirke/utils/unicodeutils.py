
# pylint: disable=unused-import
from typing import List, Tuple

import unicodedata

# Java has Character.getType(), Python has a different mechanism for similar information.
# They don't exactly match, but took the most frequent mapping as the correct mapping.
# Both are 30 and unique.  # id missing 17, but has 30.
UNICODE_CATID_MAP = {'Cc': 15,
                     'Cf': 16,
                     'Cn': 0,    # Other, unknown
                     'Co': 18,
                     'Cs': 19,
                     'Ll': 2,
                     'Lm': 4,
                     'Lo': 5,
                     'Lt': 3,
                     'Lu': 1,
                     'Mc': 8,
                     'Me': 7,
                     'Mn': 6,
                     'Nd': 9,
                     'Nl': 10,
                     'No': 11,
                     'Pc': 23,
                     'Pd': 20,
                     'Pe': 22,
                     'Pf': 30,
                     'Pi': 29,
                     'Po': 24,
                     'Ps': 21,
                     'Sc': 26,
                     'Sk': 27,
                     'Sm': 25,
                     'So': 28,
                     'Zl': 13,
                     'Zp': 14,
                     'Zs': 12}

# not used
UNICODE_CAT_LIST = ['Cc', 'Cf', 'Cn', 'Co', 'Cs', 'Ll', 'Lm', 'Lo', 'Lt', 'Lu',
                    'Mc', 'Me', 'Mn', 'Nd', 'Nl', 'No', 'Pc', 'Pd', 'Pe', 'Pf',
                    'Pi', 'Po', 'Ps', 'Sc', 'Sk', 'Sm', 'So', 'Zl', 'Zp', 'Zs']

def unicode_char_to_category_id(int_ch):
    return UNICODE_CATID_MAP.get(unicodedata.category(int_ch))


def unicode_char_to_category(int_ch):
    return unicodedata.category(int_ch)


def setup_normalize_dbs_sbs() -> Tuple[str, str]:
    """Change double-byte (full width) and single-byte (half width)
    characters to their 'natural' representations.

    Thanks John Cowen for providing the transform table."""

    alist = []  # type: List[Tuple[int, int, str]]
    with open('dict/widthmap.txt', 'rt') as fin:
        for line in fin:
            cname, incorrect_hex, norm_hex = line.strip().split(':')

            # the earlier logic to change all dbs char to sbs is
            # not correct.  We want "natural" characters, not always
            # single-byte
            """
            if incorrect_int < int('FF61', 16):  # ideographic full stop
                incorrect_int = int(incorrect_hex, 16)
                norm_int = int(norm_hex, 16)
            elif incorrect_int >= int('FFE0', 16) and \
                 incorrect_int <= int('FFE6', 16):  # cent sign, won sign
                incorrect_int = int(incorrect_hex, 16)
                norm_int = int(norm_hex, 16)
            elif incorrect_int >= int('FF61', 16):
                norm_int = int(incorrect_hex, 16)
                incorrect_int = int(norm_hex, 16)
            """
            incorrect_int = int(incorrect_hex, 16)
            norm_int = int(norm_hex, 16)

            alist.append((incorrect_int, norm_int, cname))

    alist.sort()
    incorrect_chars = []
    norm_chars = []
    for incorrect_int, norm_int, cname in alist:
        incorrect_char = chr(incorrect_int)
        norm_char = chr(norm_int)

        # print('[{}] -> [{}], {}'.format(incorrect_char, norm_char, cname))
        incorrect_chars.append(incorrect_char)
        norm_chars.append(norm_char)

    # print('incorrect_chars')
    # print(''.join(incorrect_chars))
    # print('norm_chars')
    # print(''.join(norm_chars))

    return ''.join(incorrect_chars), ''.join(norm_chars)


# Enable replacing all double-byte and single-byte characters with
# their 'natural' representations.
# INTAB, OUTTAB = setup_normalize_dbs_sbs()

# hard code those string in here to speed up processing
# it is hard to view INTAB and OUTTAB because of console handling of certain
# characters.
# Verified that the result is same as calling setup_normalize_dbs_sbs().

# pylint: disable=line-too-long
INTAB = "！＂＃＄％＆＇（）＊＋，－．／０１２３４５６７８９：；＜＝＞？＠ＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺ［＼］＾＿｀ａｂｃｄｅｆｇｈｉｊｋｌｍｎｏｐｑｒｓｔｕｖｗｘｙｚ｛｜｝～｟｠｡｢｣､･ｦｧｨｩｪｫｬｭｮｯｰｱｲｳｴｵｶｷｸｹｺｻｼｽｾｿﾀﾁﾂﾃﾄﾅﾆﾇﾈﾉﾊﾋﾌﾍﾎﾏﾐﾑﾒﾓﾔﾕﾖﾗﾘﾙﾚﾛﾜﾝﾠﾡﾢﾣﾤﾥﾦﾧﾨﾩﾪﾫﾬﾭﾮﾯﾰﾱﾲﾳﾴﾵﾶﾷﾸﾹﾺﾻﾼﾽﾾￂￃￄￅￆￇￊￋￌￍￎￏￒￓￔￕￖￗￚￛￜ￠￡￢￣￤￥￦￩￪￫￬￭￮" + '　'
# pylint: disable=line-too-long, anomalous-backslash-in-string
OUTTAB = "!\"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\]^_`abcdefghijklmnopqrstuvwxyz{|}~⦅⦆。「」、・ヲァィゥェォャュョッーアイウエオカキクケコサシスセソタチツテトナニヌネノハヒフヘホマミムメモヤユヨラリルレロワンㅤㄱㄲㄳㄴㄵㄶㄷㄸㄹㄺㄻㄼㄽㄾㄿㅀㅁㅂㅃㅄㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ¢£¬¯¦¥₩←↑→↓■○" + ' '

# the last character is a full-width space?

# print('len(INTAB) = {}'.format(len(INTAB)))
# print('len(OUTTAB) = {}'.format(len(OUTTAB)))
DBS_SBS_TRANTAB = str.maketrans(INTAB, OUTTAB)

def normalize_dbcs_sbcs(line: str) -> str:
    """Convert double-byte and single-byte to their 'normal' characters.

    The key assumption is that len(input_str) == len(output_str).
    """
    return line.translate(DBS_SBS_TRANTAB)
