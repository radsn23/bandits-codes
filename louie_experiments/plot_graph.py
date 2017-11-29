import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
from matplotlib import pyplot
from cycler import cycler


import pylab
import numpy as np
import sys
import csv
from bandit_data_format import *
from output_format import *

def plot_graph(source, dest = '', max_step = sys.maxsize, title_suffix = '', vertical_line = -1):
    '''
    Plot cumulative regret graph of several policy results against
    each other on the same data set.
    :param source: Source files as a list of tuples (file_path, name).
    :param dest: Optional, output destination image to save.
    :param max_step: Optional, max step to plot until.
    :param title_suffix: Optional, suffix for graph title.
    :param vertical_line: Optional, position to draw a vertical line.
    '''

    dpi = 80.0
    fig_width  = 1280 / dpi
    fig_height = 1024 / dpi
    figure = pyplot.figure(figsize=(fig_width, fig_height), dpi=dpi)

    for src in source:
        with open(src[0], newline='') as inf:
            reader = csv.DictReader(inf)
            line_no = 0
            if HEADER_SAMPLENUMBER not in reader.fieldnames:
                # some output files have two-level headers
                # so need to read the second line.
                reader = csv.DictReader(inf)
                line_no = 1
            sample_regret = []
            cum_regret = []
            sample_number = []
            for row in reader:
                # ignore headers for now
                line_no += 1
                if line_no >= max_step:
                    break

                sample_number.append(int(row[HEADER_SAMPLENUMBER]))
                sample_regret.append(int(row[H_ALGO_SAMPLE_REGRET]))
                cum_regret.append(int(row[H_ALGO_SAMPLE_REGRET_CUMULATIVE]))

        pyplot.plot(np.array(sample_number), np.array(cum_regret), label=src[1])


    if vertical_line > 0:
        pyplot.axvline(vertical_line, color='k', ls='dashed')

    pyplot.title('Cumulative Regret from Single Sample as a function of action timestep {}'.format(title_suffix))
    pyplot.xlabel('Timestep')
    pyplot.ylabel('Regret')
    pylab.legend(loc='upper left')

    if dest == '':
        pyplot.show()
    else:
        pylab.savefig(dest, bbox_inches='tight')

    pyplot.close(figure)


def plot_graph_average(source, dest = '', max_step = sys.maxsize, title = '', vertical_line = -1, raw_data_dest = ''):
    '''
    Plot average cumulative regret graph of several policy results against
    each other on the same data set.
    :param source: Source files as a list of list of tuples (file_path, name).
    :param dest: Optional, output destination image to save.
    :param max_step: Optional, max step to plot until.
    :param title: Optional, graph title.
    :param vertical_line: Optional, position to draw a vertical line.
    :param raw_data_dest: Optional, output destination for saving the raw data for the graph in table form
    '''

    dpi = 80.0
    fig_width  = 1280 / dpi
    fig_height = 1024 / dpi
    figure = pyplot.figure(figsize=(fig_width, fig_height), dpi=dpi)
    
    # store the raw data put into the plot so it can be written to disk; useful
    # for dealing with cycling of colors
    raw_data = {}
    
    # also include a cycler to make it easier to deal with many lines
    colors = ['r', 'g', 'b', 'y']
    linestyles = ['-', '--', ':', '-.']
    linewidths = [1, 1, 3, 3]


    color_cycler_list = []
    for c in colors:
        color_cycler_list += [c]*len(linestyles)
    
    linestyles_list = linestyles*len(colors)
    linewidths_list = linewidths*len(colors)
    pyplot.rc('axes', prop_cycle=(cycler('color', color_cycler_list) +
                           cycler('linestyle', linestyles_list) +
                           cycler('linewidth', linewidths_list)))
    
    for src in source:
        avg_cum_regret = []
        for src_i in src[0]:
            with open(src_i, newline='') as inf:
                reader = csv.DictReader(inf)
                line_no = 0
                if HEADER_SAMPLENUMBER not in reader.fieldnames:
                    # some output files have two-level headers
                    # so need to read the second line.
                    reader = csv.DictReader(inf)
                    line_no = 1
                sample_regret = []
                cum_regret = []
                sample_number = []
                for row in reader:
                    # ignore headers for now
                    line_no += 1
                    if line_no >= max_step:
                        break
                    sample_number.append(int(row[HEADER_SAMPLENUMBER]))
                    sample_regret.append(int(row[H_ALGO_SAMPLE_REGRET]))
                    cum_regret.append(int(row[H_ALGO_SAMPLE_REGRET_CUMULATIVE]))
            avg_cum_regret.append(np.array(cum_regret))
        avg_cum_regret = np.mean(np.array(avg_cum_regret), 0)

        pyplot.plot(np.array(sample_number), np.array(avg_cum_regret), label=src[1])
        raw_data[src[1]] = (np.array(sample_number), np.array(avg_cum_regret))


    if vertical_line > 0:
        pyplot.axvline(vertical_line, color='k', ls='dashed')

    pyplot.title(title)
    pyplot.xlabel('Timestep')
    pyplot.ylabel('Regret')
    pylab.legend(loc='upper left')

    if dest == '':
        pyplot.show()
    else:
        pylab.savefig(dest, bbox_inches='tight')

    pyplot.close(figure)
    
    if raw_data_dest != '':
        # Save the raw data as a csv
        with open(raw_data_dest, 'w', newline='') as out:
            raw_data_writer = csv.writer(out, delimiter=',')
            for alg in raw_data:
                raw_data_writer.writerow([alg] + raw_data[alg][1].tolist())
            


def main():
    # source files and labels to show in graph
    source = [("simulated_single_bandit_random.csv", "random"),
              ("simulated_single_bandit_thompson.csv", "thompson"),
              ("simulated_single_bandit_ucb1.csv", "ucb1")]

    plot_graph(source)

if __name__ == "__main__":
    main()