"""Build the harmonized US School Choice and Homeschooling dataset from the NCES NHES
Parent and Family Involvement in Education (PFI) public-use files, 2016 and 2019.

Input:  raw/2016_pfi_pfi_pu.csv, raw/2019_pfi_pfi_pu_pert.csv (https://nces.ed.gov/nhes/dataproducts.asp)
Output: out/pfi/data/*.csv
Value labels were checked against the 2016 and 2019 NCES codebooks; codes that differ between
waves are mapped per wave.
"""
import os, numpy as np, pandas as pd
from build_ecpp import (YES_NO, YES_NO_DK, LOCALE, INC16, INC19, INC19_DETAIL, EDU, RACE16, RACE19, FAM, PARN, WORK,
                        LANG, REGION, lab, num)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, 'raw'); OUT = os.path.join(ROOT, 'out', 'pfi', 'data')
os.makedirs(OUT, exist_ok=True)

SATIS = {1: 'Very satisfied', 2: 'Somewhat satisfied', 3: 'Somewhat dissatisfied', 4: 'Very dissatisfied'}
AGREE = {1: 'Strongly agree', 2: 'Agree', 3: 'Disagree', 4: 'Strongly disagree'}
GRADES = {1: "Mostly A's", 2: "Mostly B's", 3: "Mostly C's", 4: "Mostly D's or lower", 5: 'School does not give these grades'}
EXPECT = {1: 'Less than high school', 2: 'High school diploma', 3: 'Vocational or technical school', 4: 'Two or more years of college',
          5: "Bachelor's degree", 6: 'Graduate or professional degree'}
STYLE = {1: 'Strictly formal curriculum', 2: 'Mostly formal curriculum', 3: 'Mostly informal learning', 4: 'Always informal learning'}
MOST16 = {'A': 'School environment', 'B': 'Academic instruction', 'C': 'Religious instruction', 'D': 'Moral instruction',
          'E': 'Health problem or disability', 'F': 'Temporary illness', 'G': 'Special needs', 'H': 'Nontraditional approach',
          'I': 'Other reason'}
MOST19 = {1: 'School environment', 2: 'Academic instruction', 3: 'Religious instruction', 4: 'Moral instruction',
          5: 'Health problem or disability', 6: 'Temporary illness', 7: 'Special needs', 8: 'Nontraditional approach',
          9: 'Family life', 10: 'Other reason', 11: 'Bullying'}
ONLINE19 = {1: 'All courses online', 2: 'Half or more online', 3: 'Less than half online', 4: 'No online courses'}
IMP4 = {1: 'Not at all important', 2: 'Somewhat important', 3: 'Important', 4: 'Very important'}
REASONS = [('HSSAFETYX', 'school_environment'), ('HSDISSATX', 'dissatisfied_with_instruction'), ('HSRELGON', 'religious_instruction'),
           ('HSMORAL', 'moral_instruction'), ('HSDISABLX', 'health_problem'), ('HSILLX', 'temporary_illness'),
           ('HSSPCLNDX', 'special_needs'), ('HSALTX', 'nontraditional_approach'), ('HSFMLY', 'family_life_2019'),
           ('HSBULLY', 'bullying_2019'), ('HSOTHERX', 'other')]
CHOICE19 = [('SCHLSAFETY', 'safety'), ('LOCALE', 'convenient_location'), ('SCHLSTFQUALITY', 'staff_quality'),
            ('STUDNTPERFORM', 'academic_performance'), ('CLSSIZE', 'class_size'), ('SCHLCOST', 'cost'),
            ('RELIGSOR', 'religious_orientation'), ('AVAILCOURSE', 'curriculum_focus'), ('XTRACURRIC', 'extracurriculars'),
            ('STUDNTCHAR', 'student_body'), ('SPECALEDSERVS', 'special_education'), ('SPECALFACILTS', 'special_facilities')]


def grade16(v):
    return 'Kindergarten' if str(v) == 'K' else f'Grade {int(v)}'


def grade19(v):
    v = int(v)
    return 'Kindergarten' if v in (2, 3) else f'Grade {v - 3}'


def build(y, path):
    d = pd.read_csv(path, low_memory=False)
    o = pd.DataFrame({'survey_year': y, 'student_id': d['BASMID'].astype(str)})
    o['child_age'] = d['AGE2015' if y == 2016 else 'AGE2018']
    o['grade'] = d['ALLGRADEX'].map(grade16 if y == 2016 else grade19)
    o['child_sex'] = lab(d['CSEX'], {1: 'Male', 2: 'Female'})
    o['child_race_ethnicity'] = lab(d['RACEETHN' if y == 2016 else 'RACEETH'], RACE16 if y == 2016 else RACE19)
    o['census_region'] = lab(d['CENREG'], REGION)
    o['locale'] = lab(d['ZIPLOCL'], LOCALE)
    o['locale_group'] = o['locale'].str.split(',').str[0]
    o['parent_education'] = lab(d['PARGRADEX'], EDU)
    o['household_income'] = lab(d['TTLHHINC'], INC16 if y == 2016 else INC19)
    o['household_income_detail'] = o['household_income'] if y == 2016 else d['TTLHHINC'].map({**INC19, **INC19_DETAIL})
    o['family_type'] = lab(d['FAMILY16X' if y == 2016 else 'FAMILY19X'], FAM)
    o['parents_in_household'] = lab(d['HHPARN16X' if y == 2016 else 'HHPARN19X'], PARN)
    o['parent1_work_status'] = lab(d['PAR1EMPL'], WORK)
    o['home_language'] = lab(d['LANGUAGEX'], LANG)

    # homeschooling, NCES definition (Digest table 206.10)
    if y == 2016:
        hs = d['HMSCHLX'].isin([1, 2])
        arr = d['HMSCHLX'].map({1: 'Homeschooled full time', 2: 'Homeschooled part time, in school under 25 hours'})
    else:
        hs = (d['HMSCHARR'] == 1) | ((d['HMSCHARR'] == 2) & d['SCHLHRSWK'].isin([1, 2, 3]))
        arr = np.where(d['HMSCHARR'] == 1, 'Homeschooled full time',
                       np.where(hs, 'Homeschooled part time, in school 24 hours or less', None))
    o['homeschooled'] = np.where(hs, 'Yes', 'No')
    o['homeschool_arrangement'] = arr

    # type of schooling
    st = pd.Series('Other or unknown', index=d.index)
    if y == 2016:
        pub = d['SCPUBPRI'] == 4
        st[d['SCPUBPRI'] == 1] = 'Private, Catholic'
        st[d['SCPUBPRI'] == 2] = 'Private, other religious'
        st[d['SCPUBPRI'] == 3] = 'Private, nonreligious'
    else:
        pub = d['EDCPUB'] == 1
        st[d['EDCCAT'] == 1] = 'Private, Catholic'
        st[d['EDCREL'] == 1] = 'Private, other religious'
        st[d['EDCPRI'] == 1] = 'Private, nonreligious'
    st[pub & (d['DISTASSI'] == 2)] = 'Public, chosen'
    st[pub & (d['DISTASSI'] == 1)] = 'Public, assigned'
    st[pub & (d['SCHRTSCHL'] == 1)] = 'Public charter'
    st[hs] = 'Homeschooled'
    o['schooling_type'] = st
    o['schooling_group'] = st.str.split(',').str[0].replace({'Public charter': 'Public', 'Other or unknown': 'Other or unknown'})

    # choosing a school
    o['moved_to_attend_school'] = lab(d['SNEIGHBRX'], YES_NO)
    o['district_allows_choice'] = lab(d['SPUBCHOIX'], YES_NO_DK)
    o['considered_other_schools'] = lab(d['SCONSIDR'], YES_NO)
    o['school_is_first_choice'] = lab(d['S1STCHOI'], YES_NO)
    for v, name in CHOICE19:
        m = dict(IMP4)
        if v == 'RELIGSOR': m[5] = 'School is not religious'
        o[f'choice_reason_{name}_2019'] = lab(d[v], m) if y == 2019 else np.nan

    # school experience (enrolled students)
    for v, name in [('FCSCHOOL', 'school'), ('FCTEACHR', 'teachers'), ('FCSTDS', 'academic_standards'), ('FCORDER', 'discipline'),
                    ('FCSUPPRT', 'staff_parent_interaction')]:
        o[f'satisfaction_{name}'] = lab(d[v], SATIS)
    o['child_enjoys_school'] = lab(d['SEENJOY'], AGREE)
    o['child_grades'] = lab(d['SEGRADES'], GRADES)
    o['repeated_a_grade'] = lab(d['SEREPEAT'], YES_NO)
    o['homework_hours_week'] = num(d['FHWKHRS'])
    for v, name in [('FSSPORTX', 'attended_school_event'), ('FSVOL', 'volunteered'), ('FSMTNG', 'attended_general_meeting'),
                    ('FSPTMTNG', 'attended_pto_meeting'), ('FSATCNFN', 'attended_parent_teacher_conference'),
                    ('FSFUNDRS', 'fundraising'), ('FSCOMMTE', 'served_on_committee')]:
        o[f'parent_{name}'] = lab(d[v], YES_NO)
    if y == 2016:
        exp = d['SEFUTUREX'].where(d['SEFUTUREX'] > 0, d['HSFUTUREX'])
    else:
        exp = d['SEFUTUREX']
    o['expected_education'] = lab(exp, EXPECT)

    # homeschool details (homeschooled students)
    for v, name in REASONS:
        o[f'homeschool_reason_{name}'] = lab(d[v], YES_NO) if v in d else np.nan
    o['homeschool_most_important_reason'] = d['HSMOSTX'].astype(str).map({str(k): v for k, v in (MOST16 if y == 2016 else MOST19).items()})
    o['homeschool_teaching_style'] = lab(d['HSSTYL'], STYLE)
    o['homeschool_coop_instruction'] = lab(d['HSCOOP'], YES_NO)
    o['homeschool_organization_member'] = lab(d['HSNATL'], YES_NO)
    if y == 2016:
        o['homeschool_online_instruction'] = lab(d['HSINTNET'], YES_NO)
        o['homeschool_online_share_2019'] = np.nan
    else:
        o['homeschool_online_instruction'] = lab(d['HSINTNET'], {1: 'Yes', 2: 'Yes', 3: 'Yes', 4: 'No'})
        o['homeschool_online_share_2019'] = lab(d['HSINTNET'], ONLINE19)
    o['final_weight'] = d['FPWT'].round(4)
    rep = d[['BASMID'] + [f'FPWT{i}' for i in range(1, 81)]].rename(columns={'BASMID': 'student_id'})
    rep.insert(0, 'survey_year', y)
    rep['student_id'] = rep['student_id'].astype(str)
    return o, rep


if __name__ == '__main__':
    a, ra = build(2016, os.path.join(RAW, '2016_pfi_pfi_pu.csv'))
    b, rb = build(2019, os.path.join(RAW, '2019_pfi_pfi_pu_pert.csv'))
    df = pd.concat([a, b], ignore_index=True); rep = pd.concat([ra, rb], ignore_index=True)
    import zipfile
    for yr in (2016, 2019):   # one file per survey year keeps each file under 10 MB
        df[df.survey_year == yr].to_csv(os.path.join(OUT, f'school_choice_homeschooling_{yr}.csv'), index=False)
        tmp = os.path.join(OUT, f'replicate_weights_{yr}.csv')
        rep[rep.survey_year == yr].round(4).to_csv(tmp, index=False)
        with zipfile.ZipFile(tmp.replace('.csv', '.zip'), 'w', zipfile.ZIP_DEFLATED, compresslevel=9) as z:
            z.write(tmp, os.path.basename(tmp))
        os.remove(tmp)
    print(df.shape, rep.shape)
    print(df.groupby('survey_year').schooling_type.value_counts().to_string())
    print(df.isna().mean().round(2)[lambda s: s > 0].to_string())
