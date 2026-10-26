"""Report spectra and paired temporal changes; source SQLite is opened read-only.

Run: python analysis/spectrum_analysis.py --help
Requires numpy and matplotlib. No circuit fitting is performed.
"""
from __future__ import annotations
import argparse
import csv
import json
import re
import sqlite3
from datetime import date
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def write_csv(path, rows):
    rows = list(rows)
    if not rows:
        return
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def corrected(frequency, impedance, resistance=2000., capacitance=15e-9*470e-9/(15e-9+470e-9)):
    return impedance - (resistance + 1 / (2j*np.pi*frequency*capacitance))


def interpolate(frequency, impedance, target):
    """Interpolate real and imaginary parts versus log frequency; never extrapolate."""
    if target < frequency[0] * (1-1e-10) or target > frequency[-1] * (1+1e-10):
        raise ValueError(f'Target {target} outside measured sweep')
    return (np.interp(np.log(target), np.log(frequency), impedance.real)
            + 1j*np.interp(np.log(target), np.log(frequency), impedance.imag))


def phase_change(current, baseline):
    return float(np.angle(current * np.conj(baseline), deg=True))


def describe(values):
    values = np.asarray(values, dtype=float)
    return dict(n=len(values), mean=float(np.mean(values)),
                median=float(np.median(values)), q1=float(np.percentile(values, 25)),
                q3=float(np.percentile(values, 75)),
                sd=float(np.std(values, ddof=1)) if len(values)>1 else 0.)


def save(fig, directory, name):
    fig.savefig(directory / (name+'.png'), dpi=300, bbox_inches='tight')
    fig.savefig(directory / (name+'.pdf'), bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--database', type=Path, default=ROOT/'data/processed/circuit_fits.sqlite')
    parser.add_argument('--output', type=Path, default=ROOT/'output/spectrum_analysis')
    parser.add_argument('--banana', default='banana16', help='Representative date-sweep banana')
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    sweeps = out/'date_sweeps'
    sweeps.mkdir(exist_ok=True)
    with sqlite3.connect(args.database.resolve().as_uri()+'?mode=ro', uri=True) as conn:
        conn.row_factory = sqlite3.Row
        rows = [dict(r) for r in conn.execute('SELECT * FROM captures WHERE is_current=1 ORDER BY measurement_date,banana_id')]
    if not rows:
        raise ValueError('No current captures')
    seen = set()
    for row in rows:
        key = row['banana_id'], row['measurement_date']
        if key in seen:
            raise ValueError(f'Multiple current captures for {key}; select explicitly before analysis')
        seen.add(key)
        f = np.asarray(json.loads(row['frequency_json']), dtype=float)
        z = np.asarray(json.loads(row['real_json'])) + 1j*np.asarray(json.loads(row['imag_json']))
        order = np.argsort(f)
        f,z = f[order],z[order]
        if len(f)!=len(z) or np.any(f<=0) or not np.all(np.isfinite(z)) or np.any(np.diff(f)<=0):
            raise ValueError(f'Invalid spectrum {key}')
        row['f'], row['embedded'], row['deembedded'] = f,z,corrected(f,z)
    dates = sorted({r['measurement_date'] for r in rows})
    first,last = dates[0],dates[-1]
    reference = next(r for r in rows if r['measurement_date'] != first)
    target_low = float(reference['f'][reference['f']>1000][0])
    targets = [target_low, 180000.]
    baselines = {r['banana_id']:r for r in rows if r['measurement_date']==first}
    final_ids = {r['banana_id'] for r in rows if r['measurement_date']==last}
    cohort = set(baselines) & final_ids
    latest = {}
    for r in rows:
        latest[r['banana_id']] = r['measurement_date']
    values = []
    for r in rows:
        for domain in ('embedded','deembedded'):
            for target in targets:
                z = interpolate(r['f'],r[domain],target)
                exact = bool(np.any(np.isclose(r['f'],target,rtol=1e-9,atol=1e-7)))
                b = baselines.get(r['banana_id'])
                if b:
                    # De-embed at the measured frequencies before interpolation,
                    # matching the existing spectrum preprocessing sequence.
                    z0 = interpolate(b['f'],b[domain],target)
                    mag_change = float(100*(abs(z)/abs(z0)-1))
                    delta = phase_change(z,z0)
                else:
                    mag_change = delta = ''
                values.append(dict(capture_id=r['id'],banana=r['banana_id'],date=r['measurement_date'],
                    day=(date.fromisoformat(r['measurement_date'])-date.fromisoformat(first)).days+1,
                    domain=domain,frequency_hz=target,magnitude_ohm=float(abs(z)),phase_deg=float(np.angle(z,deg=True)),
                    magnitude_change_pct=mag_change,phase_change_deg=delta,
                    interpolated=not exact,has_day_one=b is not None,
                    endpoint_cohort=r['banana_id'] in cohort,
                    qc_status=r['qc_status'],qc_message=r['qc_message']))
    write_csv(out/'per_measurement_spectral_values.csv',values)
    summaries = []
    for population in ('all_day_one_paired','shared_endpoint_cohort'):
        for day in dates:
            for domain in ('embedded','deembedded'):
                for target in targets:
                    selected=[v for v in values if v['date']==day and v['domain']==domain and v['frequency_hz']==target
                              and v['has_day_one'] and (population=='all_day_one_paired' or v['endpoint_cohort'])]
                    for metric in ('magnitude_change_pct','phase_change_deg'):
                        if selected:
                            summaries.append(dict(population=population,date=day,day=selected[0]['day'],domain=domain,
                                frequency_hz=target,metric=metric,**describe([v[metric] for v in selected])))
    write_csv(out/'daily_change_summary.csv',summaries)
    endpoints=[]
    for kind in ('shared_final_date','last_available_date'):
        for domain in ('embedded','deembedded'):
            for target in targets:
                selected=[v for v in values if v['has_day_one'] and v['domain']==domain and v['frequency_hz']==target
                    and (v['date']==last if kind=='shared_final_date' else v['date']==latest[v['banana']])
                    and v['date']!=first]
                for metric in ('magnitude_change_pct','phase_change_deg'):
                    endpoints.append(dict(comparison=kind,domain=domain,frequency_hz=target,metric=metric,
                        **describe([v[metric] for v in selected])))
    write_csv(out/'endpoint_change_summary.csv',endpoints)
    write_csv(out/'capture_inventory.csv',[{k:r[k] for k in ('id','banana_id','measurement_date','point_count','min_frequency_hz','max_frequency_hz','qc_status','qc_message')} for r in rows])
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    colours = plt.get_cmap('viridis')(np.linspace(.05,.9,len(dates)))
    by_date = dict(zip(dates,colours))
    for banana in sorted({r['banana_id'] for r in rows},key=lambda s:int(re.search(r'\d+',s)[0])):
        fig, axes = plt.subplots(2,2,figsize=(10,6),layout='constrained')
        for r in (r for r in rows if r['banana_id']==banana):
            for col,domain in enumerate(('embedded','deembedded')):
                z=r[domain]
                axes[0,col].loglog(r['f'],abs(z),color=by_date[r['measurement_date']],label=r['measurement_date'])
                axes[1,col].semilogx(r['f'],np.angle(z,deg=True),color=by_date[r['measurement_date']])
        for col,title in enumerate(('Embedded','De-embedded')):
            axes[0,col].set(title=title,ylabel='Magnitude (ohm)')
            axes[1,col].set(xlabel='Frequency (Hz)',ylabel='Phase (degrees)')
        for ax in axes.flat:
            ax.grid(alpha=.2,which='both')
        handles,labels=axes[0,0].get_legend_handles_labels()
        fig.legend(handles,labels,loc='outside lower center',ncol=5,fontsize=8)
        fig.suptitle(f'{banana.replace("banana", "Banana ")} — measured spectra')
        save(fig,sweeps,banana)
        if banana==args.banana:
            # Save a second copy for a stable report filename.
            import shutil
            for extension in ('png','pdf'):
                shutil.copyfile(sweeps/f'{banana}.{extension}',out/f'figure_01_representative_date_sweep.{extension}')
    fig,axes=plt.subplots(2,2,figsize=(10,6),layout='constrained')
    for col,target in enumerate(targets):
        for row_index,metric in enumerate(('magnitude_change_pct','phase_change_deg')):
            ax=axes[row_index,col]
            for domain,colour in (('embedded','#277da8'),('deembedded','#e58932')):
                s=[s for s in summaries if s['population']=='shared_endpoint_cohort' and s['domain']==domain and s['frequency_hz']==target and s['metric']==metric]
                x=[s['day'] for s in s]
                ax.plot(x,[v['median'] for v in s],'o-',color=colour,label=domain.replace('deembedded','De-embedded').replace('embedded','Embedded'))
                ax.fill_between(x,[v['q1'] for v in s],[v['q3'] for v in s],color=colour,alpha=.18)
            ax.axhline(0,color='grey',lw=.7)
            ax.grid(alpha=.2)
            ax.set(title=f'{target/1000:.3f} kHz',xlabel='Measurement day',ylabel='Magnitude change (%)' if row_index==0 else 'Phase change (degrees)')
    axes[0,0].legend()
    fig.suptitle(f'Changes from day one — median and interquartile range\nShared endpoint cohort: {len(cohort)} bananas; daily counts in CSV')
    save(fig,out,'figure_02_daily_spectral_changes')
    table=['\\begin{tabular}{llrrr}','\\hline','Domain & Frequency & $n$ & Magnitude change (\\%) & Phase change (degrees) \\\\','\\hline']
    md=['| Domain | Frequency | n | Magnitude change %, median [Q1, Q3] | Phase change degrees, median [Q1, Q3] |','|---|---:|---:|---:|---:|']
    for domain in ('embedded','deembedded'):
        for target in targets:
            pair=[next(e for e in endpoints if e['comparison']=='shared_final_date' and e['domain']==domain and e['frequency_hz']==target and e['metric']==metric) for metric in ('magnitude_change_pct','phase_change_deg')]
            text=[f"{p['median']:+.1f} [{p['q1']:+.1f}, {p['q3']:+.1f}]" for p in pair]
            label='Embedded' if domain=='embedded' else 'De-embedded'
            table.append(f'{label} & {target/1000:.3f} kHz & {pair[0]["n"]} & {text[0]} & {text[1]} \\\\')
            md.append(f'| {label} | {target/1000:.3f} kHz | {pair[0]["n"]} | {text[0]} | {text[1]} |')
    table += ['\\hline','\\end{tabular}']
    (out/'endpoint_table.tex').write_text('\n'.join(table),encoding='utf-8')
    (out/'endpoint_table.md').write_text('\n'.join(md),encoding='utf-8')
    manifest=dict(database=str(args.database.resolve()),captures=len(rows),bananas=len(latest),dates=dates,
        baseline_date=first,final_date=last,endpoint_cohort=sorted(cohort),frequencies_hz=targets,
        series_resistance_ohm=2000,series_capacitance_f=15e-9*470e-9/(15e-9+470e-9),
        interpolation='Real and imaginary components linear against log frequency; correction before interpolation; no extrapolation',
        phase_difference='Wrapped signed phase difference in [-180,180] degrees',
        qc_policy='All current captures retained; flags exported; no silent filtering',
        cohorts='Daily main plot uses bananas present on both shared endpoint dates; daily sample counts may still vary. Alternate all-baseline summaries exported.',
        missing_baseline_bananas=sorted(set(latest)-set(baselines)))
    (out/'analysis_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps(manifest,indent=2))
    print('\n'+'\n'.join(md))


if __name__=='__main__':
    main()
