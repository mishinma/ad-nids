import os
import shutil
import uuid
from pathlib import Path

import ad_nids

templates_path = Path(ad_nids.__path__[0])/'templates'
with open(templates_path/'base.html', 'r') as f:
    BASE = f.read()


def merge_reports(reports, sort=True, base=True, heading=1):

    report_merged = ''
    heading = f'h{heading}'
    if sort:
        reports = sorted(reports, key=lambda x: x[0])

    for name, report in reports:
        report_merged += f'<{heading}> {name} </{heading}></br>'
        report_merged += report
        report_merged += '</br></br>'

    if base:
        report_merged = BASE.replace('{{STUFF}}', report_merged)

    return report_merged


def copy_to_static(loc_path, static_dir):
    new_name = str(uuid.uuid4()) + loc_path.suffix
    shutil.copy(loc_path, os.path.join(static_dir, new_name))
    rel_new_path = os.path.join(static_dir.name, new_name)
    return rel_new_path


def collect_plots(plot_paths, static_path):
    plots = ''
    for plot_path in plot_paths:
        static_plot_path = copy_to_static(plot_path, static_path)
        alt_text = plot_path.name[:-len(plot_path.suffix)]
        plots += f'<img src="{static_plot_path}" alt="{alt_text}"><br>\n'
    return plots