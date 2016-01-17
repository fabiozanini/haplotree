# vim: fdm=marker
'''
author:     Fabio Zanini
date:       08/12/14
content:    Plot tree of haplotypes.
'''
# Modules
import os
import argparse
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# Functions
def tree_from_json(json_file):
    '''Convert JSON into a Biopython tree'''
    from Bio import Phylo
    import json
    def node_from_json(json_data, node):
        '''Biopython Clade from json (for recursive call)'''
        for attr,val in json_data.iteritems():
            if attr == 'children':
                for sub_json in val:
                    child = Phylo.BaseTree.Clade()
                    node.clades.append(child)
                    node_from_json(sub_json, child)
            else:
                if attr == 'name':
                    node.__setattr__(attr, str(val))
                    continue

                try:
                    node.__setattr__(attr, float(val))
                except:
                    node.__setattr__(attr, val)

    try:
        with open(json_file, 'r') as infile:
            json_data = json.load(infile)
    except IOError:
        raise IOError("Cannot open "+json_file)

    tree = Phylo.BaseTree.Tree()
    node_from_json(json_data, tree.root)
    tree.root.branch_length=0.01
    return tree


def draw_tree(tree, label_func=str, do_show=True, show_confidence=True,
         # For power users
         x_offset=0, y_offset=0,
         axes=None, branch_labels=None, *args, **kwargs):
    """Plot the given tree using matplotlib (or pylab).

    The graphic is a rooted tree, drawn with roughly the same algorithm as
    draw_ascii.

    Additional keyword arguments passed into this function are used as pyplot
    options. The input format should be in the form of:
    pyplot_option_name=(tuple), pyplot_option_name=(tuple, dict), or
    pyplot_option_name=(dict).

    Example using the pyplot options 'axhspan' and 'axvline':

    >>> Phylo.draw(tree, axhspan=((0.25, 7.75), {'facecolor':'0.5'}),
    ...     axvline={'x':'0', 'ymin':'0', 'ymax':'1'})

    Visual aspects of the plot can also be modified using pyplot's own functions
    and objects (via pylab or matplotlib). In particular, the pyplot.rcParams
    object can be used to scale the font size (rcParams["font.size"]) and line
    width (rcParams["lines.linewidth"]).

    :Parameters:
        label_func : callable
            A function to extract a label from a node. By default this is str(),
            but you can use a different function to select another string
            associated with each node. If this function returns None for a node,
            no label will be shown for that node.
        do_show : bool
            Whether to show() the plot automatically.
        show_confidence : bool
            Whether to display confidence values, if present on the tree.
        axes : matplotlib/pylab axes
            If a valid matplotlib.axes.Axes instance, the phylogram is plotted
            in that Axes. By default (None), a new figure is created.
        branch_labels : dict or callable
            A mapping of each clade to the label that will be shown along the
            branch leading to it. By default this is the confidence value(s) of
            the clade, taken from the ``confidence`` attribute, and can be
            easily toggled off with this function's ``show_confidence`` option.
            But if you would like to alter the formatting of confidence values,
            or label the branches with something other than confidence, then use
            this option.
    """

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        try:
            import pylab as plt
        except ImportError:
            from Bio import MissingPythonDependencyError
            raise MissingPythonDependencyError(
                "Install matplotlib or pylab if you want to use draw.")

    import matplotlib.collections as mpcollections

    # Arrays that store lines for the plot of clades
    horizontal_linecollections = []
    vertical_linecollections = []

    # Options for displaying branch labels / confidence
    def conf2str(conf):
        if int(conf) == conf:
            return str(int(conf))
        return str(conf)
    if not branch_labels:
        if show_confidence:
            def format_branch_label(clade):
                if hasattr(clade, 'confidences'):
                    # phyloXML supports multiple confidences
                    return '/'.join(conf2str(cnf.value)
                                    for cnf in clade.confidences)
                if clade.confidence:
                    return conf2str(clade.confidence)
                return None
        else:
            def format_branch_label(clade):
                return None
    elif isinstance(branch_labels, dict):
        def format_branch_label(clade):
            return branch_labels.get(clade)
    else:
        assert callable(branch_labels), \
            "branch_labels must be either a dict or a callable (function)"
        format_branch_label = branch_labels

    # Layout

    def get_x_positions(tree):
        """Create a mapping of each clade to its horizontal position.

        Dict of {clade: x-coord}
        """
        depths = tree.depths()
        # If there are no branch lengths, assume unit branch lengths
        if not max(depths.values()):
            depths = [x_offset + depth
                      for depth in tree.depths(unit_branch_lengths=True)]
        return depths

    def get_y_positions(tree):
        """Create a mapping of each clade to its vertical position.

        Dict of {clade: y-coord}.
        Coordinates are negative, and integers for tips.
        """
        maxheight = tree.count_terminals()
        # Rows are defined by the tips
        heights = dict((tip, maxheight - i + y_offset)
                       for i, tip in enumerate(reversed(tree.get_terminals())))

        # Internal nodes: place at midpoint of children
        def calc_row(clade):
            for subclade in clade:
                if subclade not in heights:
                    calc_row(subclade)
            # Closure over heights
            heights[clade] = (heights[clade.clades[0]] +
                              heights[clade.clades[-1]]) / 2.0

        if tree.root.clades:
            calc_row(tree.root)
        return heights

    x_posns = get_x_positions(tree)
    y_posns = get_y_positions(tree)
    # The function draw_clade closes over the axes object
    if axes is None:
        fig = plt.figure()
        axes = fig.add_subplot(1, 1, 1)
    elif not isinstance(axes, plt.matplotlib.axes.Axes):
        raise ValueError("Invalid argument for axes: %s" % axes)

    def draw_clade_lines(use_linecollection=False, orientation='horizontal',
                         y_here=0, x_start=0, x_here=0, y_bot=0, y_top=0,
                         color='black', lw='.1'):
        """Create a line with or without a line collection object.

        Graphical formatting of the lines representing clades in the plot can be
        customized by altering this function.
        """
        if (use_linecollection is False and orientation == 'horizontal'):
            axes.hlines(y_here, x_start, x_here, color=color, lw=lw)
        elif (use_linecollection is True and orientation == 'horizontal'):
            horizontal_linecollections.append(mpcollections.LineCollection(
                [[(x_start, y_here), (x_here, y_here)]], color=color, lw=lw),)
        elif (use_linecollection is False and orientation == 'vertical'):
            axes.vlines(x_here, y_bot, y_top, color=color)
        elif (use_linecollection is True and orientation == 'vertical'):
            vertical_linecollections.append(mpcollections.LineCollection(
                [[(x_here, y_bot), (x_here, y_top)]], color=color, lw=lw),)

    def draw_clade(clade, x_start, color, lw):
        """Recursively draw a tree, down from the given clade."""
        x_here = x_posns[clade]
        y_here = y_posns[clade]
        # phyloXML-only graphics annotations
        if hasattr(clade, 'color') and clade.color is not None:
            color = clade.color.to_hex()
        if hasattr(clade, 'width') and clade.width is not None:
            lw = clade.width * plt.rcParams['lines.linewidth']
        # Draw a horizontal line from start to here
        draw_clade_lines(use_linecollection=True, orientation='horizontal',
                         y_here=y_here, x_start=x_start, x_here=x_here, color=color, lw=lw)
        # Add node/taxon labels
        label = label_func(clade)
        if label not in (None, clade.__class__.__name__):
            axes.text(x_here, y_here, ' %s' %
                      label, verticalalignment='center')
        # Add label above the branch (optional)
        conf_label = format_branch_label(clade)
        if conf_label:
            axes.text(0.5 * (x_start + x_here), y_here, conf_label,
                      fontsize='small', horizontalalignment='center')
        if clade.clades:
            # Draw a vertical line connecting all children
            y_top = y_posns[clade.clades[0]]
            y_bot = y_posns[clade.clades[-1]]
            # Only apply widths to horizontal lines, like Archaeopteryx
            draw_clade_lines(use_linecollection=True, orientation='vertical',
                             x_here=x_here, y_bot=y_bot, y_top=y_top, color=color, lw=lw)
            # Draw descendents
            for child in clade:
                draw_clade(child, x_here, color, lw)

    draw_clade(tree.root, 0, 'k', plt.rcParams['lines.linewidth'])

    # If line collections were used to create clade lines, here they are added
    # to the pyplot plot.
    for i in horizontal_linecollections:
        axes.add_collection(i)
    for i in vertical_linecollections:
        axes.add_collection(i)

    # Aesthetics

    if hasattr(tree, 'name') and tree.name:
        axes.set_title(tree.name)
    axes.set_xlabel('branch length')
    axes.set_ylabel('taxa')
    # Add margins around the tree to prevent overlapping the axes
    xmax = max(x_posns.values())
    axes.set_xlim(-0.05 * xmax, 1.25 * xmax)
    # Also invert the y-axis (origin at the top)
    # Add a small vertical margin, but avoid including 0 and N+1 on the y axis
    axes.set_ylim(max(y_posns.values()) + 0.8, 0.2)

    # Parse and process key word arguments as pyplot options
    for key, value in kwargs.items():
        try:
            # Check that the pyplot option input is iterable, as required
            [i for i in value]
        except TypeError:
            raise ValueError('Keyword argument "%s=%s" is not in the format '
                             'pyplot_option_name=(tuple), pyplot_option_name=(tuple, dict),'
                             ' or pyplot_option_name=(dict) '
                             % (key, value))
        if isinstance(value, dict):
            getattr(plt, str(key))(**dict(value))
        elif not (isinstance(value[0], tuple)):
            getattr(plt, str(key))(*value)
        elif (isinstance(value[0], tuple)):
            getattr(plt, str(key))(*value[0], **dict(value[1]))

    if do_show:
        plt.show()


def load_tree(filename, fmt=None):
    '''Load a tree from file'''
    from Bio import Phylo

    if fmt is None:
        fmt = filename.split('.')[-1].lower()

    if fmt == 'json':
        tree = tree_from_json(filename)
    elif fmt == 'newick':
        def set_frequency(node):
            if node.name is not None:
                try:
                    frequency = float(node.name.split(':')[-1])
                except ValueError:
                    pass
                else:
                    node.frequency = frequency

            for child in node.clades:
                set_frequency(child)

        tree = Phylo.read(filename, 'newick')
        set_frequency(tree.root)
    else:
        raise NotImplemented

    return tree


def plot_haplotype_trees(datum,
                         VERBOSE=0,
                         tree_label='root',
                         draw_legend_sizes=True,
                         draw_scale_bar=True,
                         fig_filename=None):
    '''Plot tree of minor haplotypes in a typical patient'''
    from operator import attrgetter
    import seaborn as sns
    from matplotlib import pyplot as plt

    plt.ioff()

    if VERBOSE:
        print 'Plot haplotype tree'

    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    sns.set_style('white')
    ax.grid(False)

    x_offset = 0
    y_offset = 15
    y_padding = 15

    tree = getattr(datum, tree_label)
    tree.root.branch_length = 0.01

    depths = tree.depths()
    maxdepth = max(depths.itervalues())
    mindepth = min(depths.itervalues())

    # Normalize frequencies
    freqsum = sum(leaf.frequency for leaf in tree.get_terminals())
    for leaf in tree.get_terminals():
        leaf.frequency = 1.0 * leaf.frequency / freqsum

    # Collect data for circle plot
    rmin = 5
    rmax = 150
    rfun = lambda hf: rmin + (rmax - rmin) * (hf**(0.5))
    data_circles = []
    for il, leaf in enumerate(tree.get_terminals(), 1):
        hf = leaf.frequency
        r = rfun(hf)
        y = il + y_offset
        x = depths[leaf] + x_offset
        data_circles.append((x, y, 2 * r, 'grey', 'black'))

    # Draw the tree
    draw_tree(tree,
              show_confidence=False,
              label_func=lambda x: '',
              axes=ax,
              x_offset=x_offset,
              y_offset=y_offset,
              do_show=False)

    # Add circles to the leaves
    (x, y, s, c,cs) = zip(*data_circles)
    ax.scatter(x, y, s=s, c=c, edgecolor=cs, zorder=2)
    ax.set_xlim(-0.04 * maxdepth, 1.04 * maxdepth)

    y_offset += tree.count_terminals() + y_padding

    ax.set_ylim((y_offset + y_padding, 0))
    ax.set_ylabel('')
    ax.set_yticklabels([])
    ax.set_axis_off()
    ax.xaxis.set_tick_params(labelsize=16)
    ax.set_xlabel('Genetic distance [changes / site]',
                  fontsize=16,
                  labelpad=10)


    # Draw a "legend" for sizes
    if draw_legend_sizes:
        datal = [{'hf': 0.05, 'label': '5%'},
                 {'hf': 0.20, 'label': '20%'},
                 {'hf': 1.00, 'label': '100%'}]
        ax.text(0.98 * maxdepth, 0.03 * ax.get_ylim()[0],
                'Haplotype frequency:', fontsize=16, ha='right')
        for idl, datuml in enumerate(datal):
            r = rfun(datuml['hf'])
            y = (0.07 + 0.07 * idl) * ax.get_ylim()[0]
            ax.scatter(0.85 * maxdepth, y, s=r,
                       facecolor='k',
                       edgecolor='none')
            ax.text(0.98 * maxdepth, y + 0.02 * ax.get_ylim()[0],
                    datuml['label'], ha='right',
                    fontsize=14)

    # Draw scale bar
    if draw_scale_bar:
        xbar = (0.3 + 0.3 * (len(datal) >= 9)) * maxdepth
        ybar = 0.90 * ax.get_ylim()[0]
        lbar = 0.05 * maxdepth
        lbar_label = '{:.1G}'.format(lbar)
        lbar = float(lbar_label)
        ax.plot([xbar, xbar + lbar], [ybar, ybar], lw=4, c='k')
        ax.text(xbar + 0.5 * lbar, ybar + 0.08 * ax.get_ylim()[0],
                lbar_label, fontsize=14,
                ha='center')

    #plt.tight_layout(rect=(0, -0.13, 0.98, 1))

    if fig_filename:
        fig_folder = os.path.dirname(fig_filename)

        mkdirs(fig_folder)
        fig.savefig(fig_filename)
        plt.close(fig)

    else:
        plt.show()




# Script
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Plot a haplotype tree',
                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('filename',
                        help='Filename with the tree in JSON format')
    parser.add_argument('--verbose', type=int, default=2,
                        help='Verbosity level [0-4]')
    parser.add_argument('--outputfile', default='',
                        help='Output file for the figure')


    args = parser.parse_args()

    tree = load_tree(args.filename)

    plot_haplotype_trees(tree,
                         VERBOSE=args.verbose,
                         fig_filename=args.outputfile)
