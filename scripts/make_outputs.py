"""Weighted summary tables (with jackknife standard errors) and charts for the two NHES datasets.
Estimates use the final weight; standard errors use the 80 JK1 replicate weights,
SE = sqrt(79/80 * sum((estimate_r - estimate)^2)), as described in the NHES data file user's manuals.
Run after build_ecpp.py and build_pfi.py.
"""
import os, numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FONTS = os.environ.get('FONT_DIR', os.path.join(ROOT, 'scripts', 'fonts'))
for f in os.listdir(FONTS):
    if f.endswith('.ttf'): fm.fontManager.addfont(f'{FONTS}/{f}')
SERIF, SANS = 'Fraunces', 'DM Sans 9pt'
CREAM, INK, MUTED, GRID = '#F7F3EC', '#2B2A28', '#6B6862', '#E4DED3'
TC, BLUE, OLIVE, GOLD, PLUM = '#B4532F', '#1F72B8', '#7F9A1F', '#D19A2A', '#7D4E8C'
plt.rcParams.update({'font.family': SANS, 'font.size': 13, 'axes.edgecolor': GRID, 'axes.labelcolor': MUTED,
                     'xtick.color': MUTED, 'ytick.color': MUTED, 'axes.facecolor': CREAM, 'figure.facecolor': CREAM,
                     'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True, 'grid.color': GRID,
                     'grid.linewidth': 0.8, 'axes.axisbelow': True})
R = 80


def est(df, rep, mask, num_mask, wcol='final_weight', pre='FEWT'):
    """Weighted percentage of rows in `num_mask` among rows in `mask`, with JK1 standard error."""
    m = mask.values; n = (mask & num_mask).values
    w = df[wcol].values
    p = 100 * w[n].sum() / w[m].sum()
    reps = np.array([100 * rep[f'{pre}{i}'].values[n].sum() / rep[f'{pre}{i}'].values[m].sum() for i in range(1, R + 1)])
    se = np.sqrt((R - 1) / R * ((reps - p) ** 2).sum())
    return round(p, 1), round(se, 2), int(m.sum())


def wmean(df, rep, mask, col, wcol='final_weight', pre='FEWT'):
    m = (mask & df[col].notna()).values; x = df[col].values
    w = df[wcol].values
    mu = (w[m] * x[m]).sum() / w[m].sum()
    reps = np.array([(rep[f'{pre}{i}'].values[m] * x[m]).sum() / rep[f'{pre}{i}'].values[m].sum() for i in range(1, R + 1)])
    se = np.sqrt((R - 1) / R * ((reps - mu) ** 2).sum())
    return round(mu, 2), round(se, 2), int(m.sum())


def wmedian(df, mask, col, wcol='final_weight'):
    s = df.loc[mask & df[col].notna(), [col, wcol]].sort_values(col)
    c = s[wcol].cumsum(); return round(float(s.loc[c >= c.iloc[-1] / 2, col].iloc[0]), 2)


def finish(fig, title, sub, path, source):
    fig.text(0.06, 0.94, title, family=SERIF, weight=800, size=26, color=INK, va='top')
    fig.text(0.06, 0.875, sub, size=14, color=MUTED, va='top')
    fig.text(0.06, 0.03, source, size=10.5, color=MUTED)
    fig.savefig(path, dpi=150); plt.close(fig)


def hbars(path, labels, a, b, title, sub, source, la='2016', lb='2019', xlabel='Percent', fmt='{:.0f}%', xmax=None):
    fig = plt.figure(figsize=(12, 7.5)); ax = fig.add_axes([0.3, 0.14, 0.64, 0.67])
    y = np.arange(len(labels)); h = 0.38
    if a is not None:
        a = [np.nan if v is None else v for v in a]
        ax.barh(y - h / 2, a, height=h, color=GOLD, label=la)
        for yy, v in zip(y, a):
            if not np.isnan(v): ax.text(v + 0.6, yy - h / 2, fmt.format(v), va='center', size=10.5, color=MUTED)
    ax.barh(y + (h / 2 if a is not None else 0), b, height=h if a is not None else 0.62, color=TC, label=lb)
    for yy, v in zip(y, b): ax.text(v + 0.6, yy + (h / 2 if a is not None else 0), fmt.format(v), va='center', size=10.5, color=INK)
    ax.set_yticks(y); ax.set_yticklabels(labels, size=12); ax.invert_yaxis(); ax.grid(axis='y', visible=False)
    ax.set_xlabel(xlabel)
    if xmax: ax.set_xlim(0, xmax)
    if a is not None: ax.legend(frameon=False, loc='lower right', bbox_to_anchor=(1.0, 1.0), ncol=2)
    finish(fig, title, sub, path, source)


# =====================================================================================
# 1. Early childhood care (ECPP)
# =====================================================================================
E = os.path.join(ROOT, 'out', 'ecpp'); os.makedirs(f'{E}/images', exist_ok=True)
d = pd.read_csv(f'{E}/data/early_childhood_care_2016_2019.csv', low_memory=False, dtype={'child_id': str})
rp = pd.read_csv(f'{E}/data/replicate_weights_2016_2019.zip', dtype={'child_id': str})
assert (d.child_id.values == rp.child_id.values).all()
SRC_E = 'Source: NCES NHES Early Childhood Program Participation survey, 2016 and 2019. Dataset by Montessori Theory (montessori-theory.com), CC BY 4.0.'
ALL = pd.Series(True, index=d.index)

rows = []
for y in (2016, 2019):
    Y = d.survey_year == y
    for gname, gmask in [('All children', ALL)] + [(f'Age {a}', d.child_age == a) for a in range(0, 6)]:
        for ind, col in [('Any weekly nonparental care', 'weekly_care'), ('Weekly relative care', 'weekly_relative_care'),
                         ('Weekly nonrelative care', 'weekly_nonrelative_care'), ('Weekly center-based program', 'weekly_center_program')]:
            p, se, n = est(d, rp, Y & gmask, d[col] == 'Yes')
            rows.append([y, gname, ind, p, se, n])
care = pd.DataFrame(rows, columns=['survey_year', 'group', 'indicator', 'percent', 'standard_error', 'sample_size'])
care.to_csv(f'{E}/data/weekly_care_by_age.csv', index=False)

rows = []
for y in (2016, 2019):
    Y = d.survey_year == y
    for cat in ['Center-based program', "Relative care in the child's home", 'Relative care in another home',
                'Nonrelative care in another home', "Nonrelative care in the child's home", 'Equal hours in 2 or more types of care', 'No weekly nonparental care']:
        p, se, n = est(d, rp, Y, d.main_arrangement == cat); rows.append([y, cat, p, se, n])
main = pd.DataFrame(rows, columns=['survey_year', 'main_arrangement', 'percent', 'standard_error', 'sample_size'])
main.to_csv(f'{E}/data/main_arrangement.csv', index=False)

for byv, fname in [('household_income', 'center_care_by_income.csv'), ('parent_education', 'center_care_by_parent_education.csv'),
                   ('child_race_ethnicity', 'center_care_by_race_ethnicity.csv'), ('locale_group', 'center_care_by_locale.csv'),
                   ('census_region', 'center_care_by_region.csv')]:
    rows = []
    for y in (2016, 2019):
        INC_ORDER = ['$0 to $10,000', '$10,001 to $20,000', '$20,001 to $30,000', '$30,001 to $40,000', '$40,001 to $50,000', '$50,001 to $60,000',
                     '$60,001 to $75,000', '$75,001 to $100,000', '$100,001 to $150,000', '$150,001 or more']
        EDU_ORDER = ['Less than high school', 'High school graduate or equivalent', 'Vocational/technical or some college', "Bachelor's degree", 'Graduate or professional degree']
        order = INC_ORDER if byv == 'household_income' else EDU_ORDER if byv == 'parent_education' else sorted(d[byv].dropna().unique())
        for g in order:
            M = (d.survey_year == y) & (d[byv] == g) & (d.child_age >= 3)
            p, se, n = est(d, rp, M, d.weekly_center_program == 'Yes')
            rows.append([y, g, p, se, n])
    pd.DataFrame(rows, columns=['survey_year', byv, 'percent_in_weekly_center_program_age_3_to_5', 'standard_error', 'sample_size']).to_csv(f'{E}/data/{fname}', index=False)

FACT = [('importance_reliability', 'Reliability of the arrangement'), ('importance_staff_qualifications', 'Qualifications of the staff (2019)'),
        ('importance_learning_activities', 'Learning activities'), ('importance_caregiver_availability', 'Caregiver available when needed'),
        ('importance_time_with_other_children', 'Time with other children'), ('importance_location', 'Location'),
        ('importance_cost', 'Cost'), ('importance_recommendations', 'Recommendations from family and friends'),
        ('importance_group_size', 'Number of children in the group'), ('importance_website_ratings', 'Website ratings'),
        ('importance_religious_orientation', 'Religious orientation')]
rows = []
for y in (2016, 2019):
    for col, lab in FACT:
        M = (d.survey_year == y) & d[col].notna() & (d[col] != 'Not available')
        if M.sum() == 0: continue
        p, se, n = est(d, rp, M, d[col] == 'Very important'); rows.append([y, lab, col, p, se, n])
fact = pd.DataFrame(rows, columns=['survey_year', 'factor', 'column', 'percent_very_important', 'standard_error', 'sample_size'])
fact.to_csv(f'{E}/data/what_parents_look_for.csv', index=False)

rows = []
for y in (2016, 2019):
    for arr, col in [('Center-based program', 'center_cost_weekly_per_child'), ('Relative care', 'relative_cost_weekly_per_child'),
                     ('Nonrelative care', 'nonrelative_cost_weekly_per_child')]:
        M = (d.survey_year == y) & (d[col] > 0)
        mu, se, n = wmean(d, rp, M, col)
        Mall = (d.survey_year == y) & d[col].notna()
        p0, se0, n0 = est(d, rp, Mall, d[col] == 0)
        rows.append([y, arr, mu, se, wmedian(d, M, col), p0, n])
cost = pd.DataFrame(rows, columns=['survey_year', 'arrangement', 'mean_weekly_cost_per_child_usd', 'standard_error',
                                   'median_weekly_cost_per_child_usd', 'percent_paying_nothing', 'sample_size_paying'])
d['_hourly_center'] = np.where(d.center_hours > 0, d.center_cost_weekly_per_child / d.center_hours, np.nan)
for y in (2016, 2019):
    M = (d.survey_year == y) & (d._hourly_center > 0)
    mu, se, n = wmean(d, rp, M, '_hourly_center')
    cost.loc[len(cost)] = [y, 'Center-based program, per hour', mu, se, wmedian(d, M, '_hourly_center'), np.nan, n]
cost.to_csv(f'{E}/data/weekly_cost.csv', index=False)

ACT = [('times_read_to_past_week', 'Read to 3 or more times in the past week', lambda s: s >= 3),
       ('told_story_past_week', 'Told a story 3 or more times', lambda s: s == '3 or more times'),
       ('taught_letters_words_numbers_past_week', 'Taught letters, words or numbers 3+ times', lambda s: s == '3 or more times'),
       ('sang_songs_past_week', 'Sang songs 3 or more times', lambda s: s == '3 or more times'),
       ('arts_and_crafts_past_week', 'Arts and crafts 3 or more times', lambda s: s == '3 or more times'),
       ('family_dinners_past_week', 'Family dinner 5 or more days', lambda s: s >= 5),
       ('library_visit_past_month', 'Visited a library in the past month', lambda s: s == 'Yes'),
       ('bookstore_visit_past_month', 'Visited a bookstore in the past month', lambda s: s == 'Yes')]
rows = []
for y in (2016, 2019):
    for col, lab, f in ACT:
        M = (d.survey_year == y) & (d.child_age >= 3)
        p, se, n = est(d, rp, M, f(d[col]).fillna(False)); rows.append([y, lab, col, p, se, n])
act = pd.DataFrame(rows, columns=['survey_year', 'activity', 'column', 'percent_of_children_age_3_to_5', 'standard_error', 'sample_size'])
act.to_csv(f'{E}/data/home_learning_activities.csv', index=False)

# charts
c19 = care[(care.survey_year == 2019) & care.group.str.startswith('Age')]
fig = plt.figure(figsize=(12, 6.5)); ax = fig.add_axes([0.08, 0.16, 0.86, 0.63])
ages = [f'Age {a}' for a in range(6)]; x = np.arange(6); w = 0.26
for i, (ind, col) in enumerate([('Weekly relative care', OLIVE), ('Weekly nonrelative care', GOLD), ('Weekly center-based program', TC)]):
    v = [c19[(c19.group == a) & (c19.indicator == ind)].percent.iloc[0] for a in ages]
    ax.bar(x + (i - 1) * w, v, width=w, color=col, label=ind.replace('Weekly ', '').capitalize())
    for xx, vv in zip(x, v): ax.text(xx + (i - 1) * w, vv + 1, f'{vv:.0f}', ha='center', size=9.5, color=MUTED)
ax.set_xticks(x); ax.set_xticklabels(['Under 1', '1', '2', '3', '4', '5'], size=12); ax.set_xlabel("Child's age (not yet in kindergarten)")
ax.set_ylabel('Percent of children'); ax.set_ylim(0, 100); ax.legend(frameon=False, loc='upper left', ncol=3)
finish(fig, 'Who looks after young children each week', 'Weekly care arrangements of children not yet in kindergarten, by age, 2019',
       f'{E}/images/care_by_age_2019.png', SRC_E)

f19 = fact[fact.survey_year == 2019].sort_values('percent_very_important', ascending=False)
f16 = fact[fact.survey_year == 2016].set_index('factor').percent_very_important
hbars(f'{E}/images/what_parents_look_for.png', list(f19.factor), [f16.get(k) for k in f19.factor], list(f19.percent_very_important),
      'What parents look for in child care', 'Share of parents rating each factor "very important" when choosing the main arrangement',
      SRC_E, xmax=105)

cw = cost[cost.arrangement != 'Center-based program, per hour']
fig = plt.figure(figsize=(12, 6.5)); ax = fig.add_axes([0.08, 0.16, 0.86, 0.63])
x = np.arange(3); w = 0.36
for i, (y, col) in enumerate([(2016, GOLD), (2019, TC)]):
    v = cw[cw.survey_year == y].mean_weekly_cost_per_child_usd.values
    ax.bar(x + (i - 0.5) * w, v, width=w, color=col, label=str(y))
    for xx, vv in zip(x, v): ax.text(xx + (i - 0.5) * w, vv + 3, f'${vv:.0f}', ha='center', size=11, color=INK)
ax.set_xticks(x); ax.set_xticklabels(['Center-based program', 'Relative care', 'Nonrelative care'], size=12)
ax.set_ylabel('Mean weekly cost per child, USD'); ax.legend(frameon=False)
finish(fig, 'What families pay for child care', 'Mean weekly out-of-pocket cost per child, families who pay, current dollars',
       f'{E}/images/weekly_cost.png', SRC_E)

a19 = act[act.survey_year == 2019]; a16 = act[act.survey_year == 2016].set_index('activity').percent_of_children_age_3_to_5
hbars(f'{E}/images/home_learning.png', list(a19.activity), [a16[k] for k in a19.activity], list(a19.percent_of_children_age_3_to_5),
      'Learning at home before kindergarten', 'Children aged 3 to 5 not yet in kindergarten, activities with a family member', SRC_E, xmax=105)

# cover
fig = plt.figure(figsize=(16, 8), dpi=100)
ax = fig.add_axes([0.53, 0.14, 0.43, 0.7])
v = [c19[(c19.group == a) & (c19.indicator == 'Weekly center-based program')].percent.iloc[0] for a in ages]
v2 = [c19[(c19.group == a) & (c19.indicator == 'Weekly relative care')].percent.iloc[0] for a in ages]
ax.bar(np.arange(6) - 0.2, v2, width=0.4, color=OLIVE); ax.bar(np.arange(6) + 0.2, v, width=0.4, color=TC)
ax.set_xticks(range(6)); ax.set_xticklabels(['<1', '1', '2', '3', '4', '5'], size=15); ax.set_yticks([0, 25, 50, 75]); ax.set_ylim(0, 85)
ax.set_yticklabels(['0%', '25%', '50%', '75%'], size=13)
ax.text(0.0, 1.04, 'Weekly care by age, 2019', transform=ax.transAxes, size=16, color=INK)
ax.text(2.3, v[3] + 11, 'center-based', color=TC, size=14); ax.text(-0.5, v2[0] + 6, 'relatives', color=OLIVE, size=14)
fig.text(0.05, 0.80, 'US Early Childhood', family=SERIF, weight=800, size=52, color=TC, va='top')
fig.text(0.05, 0.67, 'Care Arrangements', family=SERIF, weight=800, size=52, color=TC, va='top')
fig.text(0.05, 0.53, 'Birth to age 5, 2016 and 2019', family=SERIF, weight=600, size=30, color=INK, va='top')
n16, n19 = (d.survey_year == 2016).sum(), (d.survey_year == 2019).sum()
for i, t in enumerate([f'{len(d):,} children, NCES NHES early childhood survey', 'Relatives, nannies, day care, preschool, Head Start',
                       'Hours, weekly cost, what parents look for, learning at home']):
    fig.text(0.05, 0.40 - i * 0.065, t, size=18, color=INK, va='top')
fig.text(0.05, 0.08, 'Montessori Theory  ·  montessori-theory.com  ·  CC BY 4.0', size=15, color=MUTED, family=SERIF, weight=600)
fig.savefig(f'{E}/images/cover.png', dpi=100); plt.close(fig)

print(care[care.group == 'All children'].to_string(index=False)); print(main.to_string(index=False)); print(cost.to_string(index=False))
print(fact.to_string(index=False))

# =====================================================================================
# 2. School choice and homeschooling (PFI)
# =====================================================================================
P = os.path.join(ROOT, 'out', 'pfi'); os.makedirs(f'{P}/images', exist_ok=True)
s = pd.concat([pd.read_csv(f'{P}/data/school_choice_homeschooling_{y}.csv', low_memory=False, dtype={'student_id': str}) for y in (2016, 2019)], ignore_index=True)
rq = pd.concat([pd.read_csv(f'{P}/data/replicate_weights_{y}.zip', dtype={'student_id': str}) for y in (2016, 2019)], ignore_index=True)
assert (s.student_id.values == rq.student_id.values).all()
SRC_P = 'Source: NCES NHES Parent and Family Involvement in Education survey, 2016 and 2019. Dataset by Montessori Theory (montessori-theory.com), CC BY 4.0.'
K = dict(pre='FPWT')
U = (s.child_age >= 5) & (s.child_age <= 17)            # NCES universe for homeschooling estimates
TYPES = ['Public, assigned', 'Public, chosen', 'Public charter', 'Private, Catholic', 'Private, other religious', 'Private, nonreligious', 'Homeschooled', 'Other or unknown']

rows = []
for y in (2016, 2019):
    for t in TYPES:
        p, se, n = est(s, rq, U & (s.survey_year == y), s.schooling_type == t, **K); rows.append([y, t, p, se, n])
stype = pd.DataFrame(rows, columns=['survey_year', 'schooling_type', 'percent_of_students_age_5_to_17', 'standard_error', 'sample_size'])
stype.to_csv(f'{P}/data/schooling_type.csv', index=False)

rows = []
for y in (2016, 2019):
    Y = U & (s.survey_year == y)
    p, se, n = est(s, rq, Y, s.homeschooled == 'Yes', **K)
    wsum = s.loc[Y & (s.homeschooled == 'Yes'), 'final_weight'].sum()
    rows.append([y, 'All students', 'All students', p, se, round(wsum / 1000), n])
    for byv in ['child_race_ethnicity', 'census_region', 'locale_group', 'parent_education', 'household_income', 'grade']:
        for g in s.loc[Y, byv].dropna().unique():
            M = Y & (s[byv] == g); p, se, n = est(s, rq, M, s.homeschooled == 'Yes', **K)
            rows.append([y, byv, g, p, se, round(s.loc[M & (s.homeschooled == 'Yes'), 'final_weight'].sum() / 1000), n])
hsr = pd.DataFrame(rows, columns=['survey_year', 'characteristic', 'group', 'percent_homeschooled', 'standard_error', 'homeschooled_thousands', 'sample_size'])
hsr.to_csv(f'{P}/data/homeschooling_rate.csv', index=False)

REAS = [('homeschool_reason_school_environment', 'Concern about school environment'), ('homeschool_reason_moral_instruction', 'To provide moral instruction'),
        ('homeschool_reason_family_life_2019', 'Emphasize family life together (2019)'), ('homeschool_reason_dissatisfied_with_instruction', 'Dissatisfied with academic instruction'),
        ('homeschool_reason_religious_instruction', 'To provide religious instruction'), ('homeschool_reason_nontraditional_approach', 'Nontraditional approach to education'),
        ('homeschool_reason_special_needs', 'Child has special needs'), ('homeschool_reason_health_problem', 'Child has a health problem'),
        ('homeschool_reason_other', 'Other reason'), ('homeschool_reason_temporary_illness', 'Temporary illness')]
rows = []
for y in (2016, 2019):
    H = U & (s.survey_year == y) & (s.homeschooled == 'Yes')
    for col, lab in REAS:
        M = H & s[col].notna()
        if M.sum() == 0: continue
        p, se, n = est(s, rq, M, s[col] == 'Yes', **K); rows.append([y, 'Reason cited', lab, p, se, n])
    for r in s.loc[H, 'homeschool_most_important_reason'].dropna().unique():
        M = H & s.homeschool_most_important_reason.notna()
        p, se, n = est(s, rq, M, s.homeschool_most_important_reason == r, **K); rows.append([y, 'Most important reason', r, p, se, n])
    for col in ['homeschool_teaching_style']:
        for r in s.loc[H, col].dropna().unique():
            M = H & s[col].notna(); p, se, n = est(s, rq, M, s[col] == r, **K); rows.append([y, 'Teaching style', r, p, se, n])
    for col, lab in [('homeschool_coop_instruction', 'Some instruction from a homeschool group or co-op'), ('homeschool_online_instruction', 'Some online instruction'),
                     ('homeschool_organization_member', 'Member of a homeschool organization')]:
        M = H & s[col].notna(); p, se, n = est(s, rq, M, s[col] == 'Yes', **K); rows.append([y, 'Practice', lab, p, se, n])
hsreas = pd.DataFrame(rows, columns=['survey_year', 'measure', 'category', 'percent_of_homeschooled_students', 'standard_error', 'sample_size'])
hsreas.to_csv(f'{P}/data/homeschooling_reasons_and_practices.csv', index=False)

rows = []
SAT = [('satisfaction_school', 'School'), ('satisfaction_teachers', 'Teachers'), ('satisfaction_academic_standards', 'Academic standards'),
       ('satisfaction_discipline', 'Order and discipline'), ('satisfaction_staff_parent_interaction', 'Staff and parent interaction')]
for y in (2016, 2019):
    for grp in ['Public, assigned', 'Public, chosen', 'Public charter', 'Private, Catholic', 'Private, other religious', 'Private, nonreligious']:
        for col, lab in SAT:
            M = (s.survey_year == y) & (s.schooling_type == grp) & s[col].notna()
            p, se, n = est(s, rq, M, s[col] == 'Very satisfied', **K); rows.append([y, grp, lab, p, se, n])
sat = pd.DataFrame(rows, columns=['survey_year', 'schooling_type', 'aspect', 'percent_very_satisfied', 'standard_error', 'sample_size'])
sat.to_csv(f'{P}/data/parent_satisfaction.csv', index=False)

CH = [c for c in s.columns if c.startswith('choice_reason_')]
rows = []
for grp in ['All who chose', 'Public, chosen', 'Public charter', 'Private, Catholic', 'Private, other religious', 'Private, nonreligious']:
    for col in CH:
        M = (s.survey_year == 2019) & s[col].notna() & (s[col] != 'School is not religious')
        if grp != 'All who chose': M &= s.schooling_type == grp
        p, se, n = est(s, rq, M, s[col] == 'Very important', **K)
        rows.append([2019, grp, col.replace('choice_reason_', '').replace('_2019', '').replace('_', ' ').capitalize(), p, se, n])
chs = pd.DataFrame(rows, columns=['survey_year', 'schooling_type', 'reason', 'percent_very_important', 'standard_error', 'sample_size'])
chs.to_csv(f'{P}/data/school_choice_reasons_2019.csv', index=False)

INV = {'parent_attended_school_event': 'Attended a school or class event', 'parent_volunteered': 'Volunteered at the school',
       'parent_attended_general_meeting': 'Attended a general school meeting', 'parent_attended_pto_meeting': 'Attended a PTO or PTA meeting',
       'parent_attended_parent_teacher_conference': 'Attended a parent-teacher conference', 'parent_fundraising': 'Took part in fundraising',
       'parent_served_on_committee': 'Served on a school committee'}
rows = []
for y in (2016, 2019):
    for grp in ['All enrolled students', 'Public', 'Private']:
        for col in INV:
            M = (s.survey_year == y) & s[col].notna()
            if grp != 'All enrolled students': M &= s.schooling_group == grp
            p, se, n = est(s, rq, M, s[col] == 'Yes', **K)
            rows.append([y, grp, INV[col], p, se, n])
inv = pd.DataFrame(rows, columns=['survey_year', 'schooling_group', 'activity', 'percent_of_parents', 'standard_error', 'sample_size'])
inv.to_csv(f'{P}/data/parent_involvement.csv', index=False)

# charts
lab = ['Public, assigned', 'Public, chosen', 'Public charter', 'Private, Catholic', 'Private, other religious', 'Private, nonreligious', 'Homeschooled']
t16 = stype[stype.survey_year == 2016].set_index('schooling_type').percent_of_students_age_5_to_17
t19 = stype[stype.survey_year == 2019].set_index('schooling_type').percent_of_students_age_5_to_17
hbars(f'{P}/images/schooling_type.png', lab, [t16[k] for k in lab], [t19[k] for k in lab], 'How American children go to school',
      'Students aged 5 to 17, kindergarten to grade 12 or equivalent', SRC_P, fmt='{:.1f}%', xmax=82)

r19 = hsreas[(hsreas.survey_year == 2019) & (hsreas.measure == 'Reason cited')].sort_values('percent_of_homeschooled_students', ascending=False)
r16 = hsreas[(hsreas.survey_year == 2016) & (hsreas.measure == 'Reason cited')].set_index('category').percent_of_homeschooled_students
hbars(f'{P}/images/homeschool_reasons.png', list(r19.category), [r16.get(k) for k in r19.category], list(r19.percent_of_homeschooled_students),
      'Why families homeschool', 'Share of homeschooled students whose parents cite each reason (several possible)', SRC_P, xmax=100)

s19 = sat[(sat.survey_year == 2019) & (sat.aspect == 'School')]
s16 = sat[(sat.survey_year == 2016) & (sat.aspect == 'School')].set_index('schooling_type').percent_very_satisfied
hbars(f'{P}/images/satisfaction.png', list(s19.schooling_type), [s16[k] for k in s19.schooling_type], list(s19.percent_very_satisfied),
      'How satisfied parents are with the school', 'Share of parents "very satisfied" with their child\'s school, by type of school', SRC_P, xmax=100)

c19 = chs[chs.schooling_type == 'All who chose'].sort_values('percent_very_important', ascending=False)
hbars(f'{P}/images/school_choice_reasons.png', list(c19.reason), None, list(c19.percent_very_important), 'Why parents choose a school',
      'Share rating each reason "very important", parents who chose their child\'s school, 2019', SRC_P, xmax=100)

hs_all = hsr[hsr.characteristic == 'All students'].set_index('survey_year')
fig = plt.figure(figsize=(16, 8), dpi=100)
ax = fig.add_axes([0.55, 0.14, 0.41, 0.7])
order = ['Public, assigned', 'Public, chosen', 'Public charter', 'Private, Catholic', 'Private, other religious', 'Private, nonreligious', 'Homeschooled']
cols = [TC, '#D9876A', '#E8B7A3', BLUE, '#6FA3D6', '#A9C8E8', OLIVE]
left = 0
for k, c in zip(order, cols):
    ax.barh([0], [t19[k]], left=left, color=c, height=0.5); left += t19[k]
ax.set_xlim(0, 100); ax.set_ylim(-0.6, 2.2); ax.axis('off')
labels = [('Public, assigned', t19['Public, assigned'], TC), ('Public by choice or charter', t19['Public, chosen'] + t19['Public charter'], '#D9876A'),
          ('Private', t19['Private, Catholic'] + t19['Private, other religious'] + t19['Private, nonreligious'], BLUE), ('Homeschooled', t19['Homeschooled'], OLIVE)]
for i, (l, v, c) in enumerate(labels):
    ax.text(0, 1.95 - i * 0.33, f'{v:.0f}%', color=c, size=24, family=SERIF, weight=800, va='center')
    ax.text(14, 1.95 - i * 0.33, l, color=INK, size=17, va='center')
ax.text(0, -0.45, 'Students aged 5 to 17, 2019', size=14, color=MUTED)
fig.text(0.05, 0.80, 'US School Choice', family=SERIF, weight=800, size=52, color=TC, va='top')
fig.text(0.05, 0.67, 'and Homeschooling', family=SERIF, weight=800, size=52, color=TC, va='top')
fig.text(0.05, 0.53, 'Kindergarten to grade 12, 2016 and 2019', family=SERIF, weight=600, size=28, color=INK, va='top')
for i, t in enumerate([f'{len(s):,} students, NCES NHES parent survey', 'Public, charter, private and homeschool',
                       'Why parents choose, why they homeschool, satisfaction']):
    fig.text(0.05, 0.40 - i * 0.065, t, size=18, color=INK, va='top')
fig.text(0.05, 0.08, 'Montessori Theory  ·  montessori-theory.com  ·  CC BY 4.0', size=15, color=MUTED, family=SERIF, weight=600)
fig.savefig(f'{P}/images/cover.png', dpi=100); plt.close(fig)

print(stype.to_string(index=False)); print(hs_all.to_string())
print(hsreas[hsreas.measure == 'Reason cited'].to_string(index=False))
print(chs[chs.schooling_type == 'All who chose'].to_string(index=False))
