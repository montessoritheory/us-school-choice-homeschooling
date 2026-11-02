"""Build the harmonized US Early Childhood Care Arrangements dataset from the NCES NHES
Early Childhood Program Participation (ECPP) public-use files, 2016 and 2019.

Input:  raw/2016_ecpp_ecpp_pu.csv, raw/2019_ecpp_ecpp_pu_pert.csv (https://nces.ed.gov/nhes/dataproducts.asp)
Output: out/ecpp/data/*.csv
Every value label below was checked against the 2016 and 2019 NCES codebooks; where a code
means different things in the two waves, the map is given per wave.
"""
import os, numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, 'raw'); OUT = os.path.join(ROOT, 'out', 'ecpp', 'data')
os.makedirs(OUT, exist_ok=True)

YES_NO = {1: 'Yes', 2: 'No'}
YES_NO_DK = {1: 'Yes', 2: 'No', 3: "Don't know"}
IMPORTANCE = {1: 'Not at all important', 2: 'A little important', 3: 'Somewhat important', 4: 'Very important'}
TIMES3 = {1: 'Not at all', 2: '1 or 2 times', 3: '3 or more times'}
LOCALE = {11: 'City, large', 12: 'City, midsize', 13: 'City, small', 21: 'Suburb, large', 22: 'Suburb, midsize',
          23: 'Suburb, small', 31: 'Town, fringe', 32: 'Town, distant', 33: 'Town, remote', 41: 'Rural, fringe',
          42: 'Rural, distant', 43: 'Rural, remote'}
ARR = {1: "Relative care in the child's home", 2: 'Relative care in another home', 3: "Nonrelative care in the child's home",
       4: 'Nonrelative care in another home', 5: 'Center-based program', 6: 'Equal hours in 2 or more types of care',
       -1: 'No weekly nonparental care'}
INC16 = {1: '$0 to $10,000', 2: '$10,001 to $20,000', 3: '$20,001 to $30,000', 4: '$30,001 to $40,000', 5: '$40,001 to $50,000',
         6: '$50,001 to $60,000', 7: '$60,001 to $75,000', 8: '$75,001 to $100,000', 9: '$100,001 to $150,000', 10: '$150,001 or more'}
INC19 = {**{k: v for k, v in INC16.items() if k < 10}, 10: '$150,001 or more', 11: '$150,001 or more', 12: '$150,001 or more'}
INC19_DETAIL = {10: '$150,001 to $200,000', 11: '$200,001 to $250,000', 12: '$250,001 or more'}
EDU = {1: 'Less than high school', 2: 'High school graduate or equivalent', 3: 'Vocational/technical or some college',
       4: "Bachelor's degree", 5: 'Graduate or professional degree'}
RACE16 = {1: 'White, non-Hispanic', 2: 'Black, non-Hispanic', 3: 'Hispanic', 4: 'Other or multiple races, non-Hispanic'}
RACE19 = {1: 'White, non-Hispanic', 2: 'Black, non-Hispanic', 3: 'Hispanic', 4: 'Other or multiple races, non-Hispanic',
          5: 'Other or multiple races, non-Hispanic'}
FAM = {1: 'Two parents and sibling(s)', 2: 'Two parents, no sibling', 3: 'One parent and sibling(s)', 4: 'One parent, no sibling', 5: 'Other'}
PARN = {1: 'Two parents', 2: 'Mother only', 3: 'Father only', 4: 'Nonparent guardian(s)'}
WORK = {1: 'Working 35+ hours a week', 2: 'Working under 35 hours a week', 3: 'Looking for work', 4: 'Not in the labor force'}
LANG = {1: 'English (both or only parent)', 2: 'English (one of two parents)', 3: 'Non-English language'}
REGION = {1: 'Northeast', 2: 'South', 3: 'Midwest', 4: 'West'}
PLACE = {1: 'Place of worship', 2: 'Public school', 3: 'Private school', 4: 'College or university', 5: 'Community center',
         7: 'Own building, office or storefront', 8: 'Some other place', 9: 'Center, location not specified', 10: 'A home'}
UNIT16 = {1: 'hour', 2: 'day', 3: 'week', 4: 'month', 5: 'year', 6: 'every 2 weeks', 7: 'other'}
UNIT19 = {1: 'hour', 2: 'day', 3: 'week', 4: 'every 2 weeks', 5: 'month', 6: 'year', 7: 'other'}
REASON16 = {1: 'Care while a parent works or studies', 2: 'Prepare the child for school', 3: 'Cultural or language learning',
            4: 'Time for errands or free time', 5: 'Some other reason', 6: 'Did not have care in the past year'}
REASON19 = {1: 'Did not have care in the past year', 2: 'Care while a parent works or studies', 3: 'Prepare the child for school',
            4: 'Cultural or language learning', 5: 'Time for errands or free time', 6: 'Some other reason'}
DIFF16 = {1: 'Did not look for care', 2: 'No difficulty', 3: 'A little difficulty', 4: 'Some difficulty', 5: 'A lot of difficulty',
          6: 'Did not find the program wanted'}
DIFF19 = {-1: 'Did not look for care', 1: 'No difficulty', 2: 'A little difficulty', 3: 'Some difficulty', 4: 'A lot of difficulty',
          5: 'Did not find the program wanted'}
WHY = {1: 'Cost', 2: 'Location', 3: 'Quality', 4: 'No open slots', 5: 'Special needs program needed', 6: 'Other reason',
       7: 'Did not look for care', 8: 'Wanted a particular type of program', 9: 'Specific hours or schedule',
       10: 'Financial assistance problems', 11: 'Age requirements', 12: 'Multiple reasons'}
LETTERS = {1: 'None', 2: 'Some', 3: 'Most', 4: 'All'}
COUNT = {1: 'Cannot count', 2: 'Up to 5', 3: 'Up to 10', 4: 'Up to 20', 5: 'Up to 50', 6: 'Up to 100 or more'}
CTYPE16 = {1: 'Day care', 2: 'Preschool', 3: 'Prekindergarten'}


def lab(s, m):
    return s.map(m)


def num(s):
    s = pd.to_numeric(s, errors='coerce')
    return s.where(s >= 0)


def weekly(amount, unit, hours, days, n_children):
    a = num(amount)
    f = pd.Series(np.nan, index=a.index)
    f[unit == 'hour'] = hours[unit == 'hour']
    f[unit == 'day'] = days[unit == 'day']
    f[unit == 'week'] = 1
    f[unit == 'every 2 weeks'] = 0.5
    f[unit == 'month'] = 12 / 52
    f[unit == 'year'] = 1 / 52
    w = a * f
    w[a == 0] = 0.0
    n = num(n_children).clip(upper=5).fillna(1)
    return (w / n).round(2)


def build(year, path):
    d = pd.read_csv(path, low_memory=False)
    y = year
    unit = UNIT16 if y == 2016 else UNIT19
    o = pd.DataFrame()
    o['survey_year'] = y
    o['child_id'] = d['BASMID'].astype(str)
    o['survey_year'] = y
    o['child_age'] = d['AGE2015' if y == 2016 else 'AGE2018']
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
    # care arrangements
    o['any_care_arrangement'] = lab(d['ANYCAREX'], {1: 'Yes', 2: 'No'})
    wk = (d['RCWEEK'] == 1) | (d['NCWEEK'] == 1) | (d['CPWEEKX'] == 1)
    o['weekly_care'] = np.where(wk, 'Yes', 'No')
    o['weekly_relative_care'] = np.where(d['RCWEEK'] == 1, 'Yes', 'No')
    o['weekly_nonrelative_care'] = np.where(d['NCWEEK'] == 1, 'Yes', 'No')
    o['weekly_center_program'] = np.where(d['CPWEEKX'] == 1, 'Yes', 'No')
    o['main_arrangement'] = lab(d['MOSTHRSX'], ARR)
    o['relative_care'] = lab(d['RCNOW'], YES_NO)
    o['nonrelative_care'] = lab(d['NCNOW'], YES_NO)
    o['center_program'] = lab(d['CPNNOWX'], YES_NO)
    o['relative_care_hours'] = num(d['RCHRS'])
    o['nonrelative_care_hours'] = num(d['NCHRS'])
    o['center_hours'] = num(d['CPHRS'])
    o['total_care_hours'] = num(d['CAREHOURX'])
    o['center_days'] = num(d['CPDAYS'])
    o['center_type_2016'] = lab(d['CPTYPE'], CTYPE16) if y == 2016 else np.nan
    o['center_head_start'] = lab(d['CPHEADST'], YES_NO_DK)
    o['ever_head_start'] = lab(d['PCEVRHDX'], YES_NO_DK)
    o['center_location'] = lab(d['CPPLACEX'], PLACE)
    o['center_at_parent_workplace'] = lab(d['CPWORK'], YES_NO)
    o['center_run_by_religious_group_2016'] = lab(d['CPSPRLG'], YES_NO) if y == 2016 else np.nan
    o['center_teaches_religion_2019'] = lab(d['CPSPRLG'], YES_NO) if y == 2019 else np.nan
    # costs, as reported, then a weekly cost per child
    for p, name, hrs, days in [('CP', 'center', 'CPHRS', 'CPDAYS'), ('RC', 'relative', 'RCHRS', 'RCDAYS'), ('NC', 'nonrelative', 'NCHRS', 'NCDAYS')]:
        u = d[p + 'UNIT'].map(unit)
        o[f'{name}_cost_amount'] = num(d[p + 'COST'])
        o[f'{name}_cost_unit'] = u
        o[f'{name}_cost_children_covered'] = num(d[p + 'CSTHNX'])
        o[f'{name}_cost_weekly_per_child'] = weekly(d[p + 'COST'], u, num(d[hrs]), num(d[days]), d[p + 'CSTHNX'])
    # choosing care
    o['good_choices_available'] = lab(d['PPCHOIC'], YES_NO_DK)
    o['difficulty_finding_care'] = lab(d['PPDIFCLT'], DIFF16 if y == 2016 else DIFF19)
    o['main_difficulty'] = lab(d['WHYDIFCLT'], WHY)
    o['main_reason_for_care'] = lab(d['MAINRESN'], REASON16 if y == 2016 else REASON19)
    for v, name in [('DCLOA', 'location'), ('DCOST', 'cost'), ('DRELY', 'reliability'), ('DLERN', 'learning_activities'),
                    ('DCHIL', 'time_with_other_children'), ('DHROP', 'caregiver_availability'), ('DNBGRP', 'group_size'),
                    ('DRTWEB', 'website_ratings'), ('DRECFAM', 'recommendations'), ('DRELOR', 'religious_orientation'),
                    ('DQUAL', 'staff_qualifications')]:
        m = dict(IMPORTANCE)
        if y == 2019 and v == 'DRTWEB': m[5] = 'Not available'
        if y == 2019 and v == 'DRECFAM': m[5] = 'Not available'
        o[f'importance_{name}'] = lab(d[v], m) if v in d else np.nan
    # home learning activities
    o['times_read_to_past_week'] = num(d['FOREADTOX']).clip(upper=20)
    o['told_story_past_week'] = lab(d['FOSTORYX'], TIMES3)
    o['taught_letters_words_numbers_past_week'] = lab(d['FOWORDSX'], TIMES3)
    o['sang_songs_past_week'] = lab(d['FOSANG'], TIMES3)
    o['arts_and_crafts_past_week'] = lab(d['FOCRAFTSX'], TIMES3)
    o['family_dinners_past_week'] = num(d['FODINNERX'])
    o['library_visit_past_month'] = lab(d['FOLIBRAY'], YES_NO)
    o['bookstore_visit_past_month'] = lab(d['FOBOOKST'], YES_NO)
    o['books_owned'] = num(d['HABOOKS'])
    # early skills (asked for children aged 3 and over)
    o['recognizes_letters'] = lab(d['DPLETTER'], LETTERS)
    o['counts_up_to'] = lab(d['DPCOUNT'], COUNT)
    o['writes_first_name'] = lab(d['DPNAME'], YES_NO)
    o['final_weight'] = d['FEWT'].round(4)
    rep = d[['BASMID'] + [f'FEWT{i}' for i in range(1, 81)]].copy()
    rep.insert(0, 'survey_year', y)
    rep = rep.rename(columns={'BASMID': 'child_id'})
    rep['child_id'] = rep['child_id'].astype(str)
    return o, rep


if __name__ == '__main__':
    a, ra = build(2016, os.path.join(RAW, '2016_ecpp_ecpp_pu.csv'))
    b, rb = build(2019, os.path.join(RAW, '2019_ecpp_ecpp_pu_pert.csv'))
    df = pd.concat([a, b], ignore_index=True)
    rep = pd.concat([ra, rb], ignore_index=True)
    df.to_csv(os.path.join(OUT, 'early_childhood_care_2016_2019.csv'), index=False)
    import zipfile
    tmp = os.path.join(OUT, 'replicate_weights_2016_2019.csv')
    rep.round(4).to_csv(tmp, index=False)
    with zipfile.ZipFile(tmp.replace('.csv', '.zip'), 'w', zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        z.write(tmp, os.path.basename(tmp))
    os.remove(tmp)
    print(df.shape, rep.shape)
    print(df.isna().mean().round(2).to_string())
