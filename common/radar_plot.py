import matplotlib.pyplot as plt
import numpy as np

from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path
from matplotlib.projections import register_projection
from matplotlib.projections.polar import PolarAxes
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D
from matplotlib.ticker import FuncFormatter

def radar_factory(num_vars, frame='circle'):
    """
    Create a radar chart with `num_vars` axes.

    This function creates a RadarAxes projection and registers it.

    Parameters
    ----------
    num_vars : int
        Number of variables for radar chart.
    frame : {'circle', 'polygon'}
        Shape of frame surrounding axes.

    """
    # calculate evenly-spaced axis angles
    theta = np.linspace(0, 2*np.pi, num_vars, endpoint=False)

    class RadarTransform(PolarAxes.PolarTransform):

        def transform_path_non_affine(self, path):
            # Paths with non-unit interpolation steps correspond to gridlines,
            # in which case we force interpolation (to defeat PolarTransform's
            # autoconversion to circular arcs).
            if path._interpolation_steps > 1:
                path = path.interpolated(num_vars)
            return Path(self.transform(path.vertices), path.codes)

    class RadarAxes(PolarAxes):

        name = 'radar'
        PolarTransform = RadarTransform

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # rotate plot such that the first axis is at the top
            self.set_theta_zero_location('N')

        def fill(self, *args, closed=True, **kwargs):
            """Override fill so that line is closed by default"""
            return super().fill(closed=closed, *args, **kwargs)

        def plot(self, *args, **kwargs):
            """Override plot so that line is closed by default"""
            lines = super().plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)

        def _close_line(self, line):
            x, y = line.get_data()
            # FIXME: markers at x[0], y[0] get doubled-up
            if x[0] != x[-1]:
                x = np.append(x, x[0])
                y = np.append(y, y[0])
                line.set_data(x, y)

        def set_varlabels(self, labels):
            self.set_thetagrids(np.degrees(theta), labels)

        def _gen_axes_patch(self):
            # The Axes patch must be centered at (0.5, 0.5) and of radius 0.5
            # in axes coordinates.
            if frame == 'circle':
                return Circle((0.5, 0.5), 0.5)
            elif frame == 'polygon':
                return RegularPolygon((0.5, 0.5), num_vars,
                                      radius=.5, edgecolor="k")
            else:
                raise ValueError("Unknown value for 'frame': %s" % frame)

        def _gen_axes_spines(self):
            if frame == 'circle':
                return super()._gen_axes_spines()
            elif frame == 'polygon':
                # spine_type must be 'left'/'right'/'top'/'bottom'/'circle'.
                spine = Spine(axes=self,
                              spine_type='circle',
                              path=Path.unit_regular_polygon(num_vars))
                # unit_regular_polygon gives a polygon of radius 1 centered at
                # (0, 0) but we want a polygon of radius 0.5 centered at (0.5,
                # 0.5) in axes coordinates.
                spine.set_transform(Affine2D().scale(.5).translate(.5, .5)
                                    + self.transAxes)
                return {'polar': spine}
            else:
                raise ValueError("Unknown value for 'frame': %s" % frame)

    register_projection(RadarAxes)
    return theta


def example_data():
    # The following data is from the Denver Aerosol Sources and Health study.
    # See doi:10.1016/j.atmosenv.2008.12.017
    #
    # The data are pollution source profile estimates for five modeled
    # pollution sources (e.g., cars, wood-burning, etc) that emit 7-9 chemical
    # species. The radar charts are experimented with here to see if we can
    # nicely visualize how the modeled source profiles change across four
    # scenarios:
    #  1) No gas-phase species present, just seven particulate counts on
    #     Sulfate
    #     Nitrate
    #     Elemental Carbon (EC)
    #     Organic Carbon fraction 1 (OC)
    #     Organic Carbon fraction 2 (OC2)
    #     Organic Carbon fraction 3 (OC3)
    #     Pyrolyzed Organic Carbon (OP)
    #  2)Inclusion of gas-phase specie carbon monoxide (CO)
    #  3)Inclusion of gas-phase specie ozone (O3).
    #  4)Inclusion of both gas-phase species is present...
    # data = [
    #     ['Lorenz', 'Silice', 'Nagoya', 'Osaka', 'Fukushima', 'Picophytoplankton', 'Nanophytoplankton'],
    #     ('Arima', [
    #         [0.143/0.143, 0.5/0.5, 0.390,  0.338, 0.599, 2.146/2.146, 1.351/1.351],
    #         [0.136/0.143, 0.428/0.5, 0.04,  0.02, 0.01, 1.715/2.146, 1.307/1.351]]),
    #     ('ETS', [
    #         [0.163/0.163, 0.494/0.492, 0.377,  0.320, 0.580, 1.429/1.429, 1.399/1.399],
    #         [0.155/0.163, 0.429/0.492, 0.04,  0.01, 0.12, 1.186/1.429, 1.345/1.399]]),
    #     ('Theta', [
    #         [0.181/0.181, 0.507/0.507, 0.398,  0.328, 0.621, 1.319/1.319, 1.438/1.438],
    #         [0.171/0.171, 0.435/0.507, 0.05,  0.02, 0.618, 1.120/1.319, 1.351/1.438]]),
    # ]
    data = [
        [ 'Nagoya', 'Osaka', 'Fukushima', 'Picophytoplankton', 'Nanophytoplankton','Lorenz', 'BOLD'],
        ('Arima', [
            [ 1-1.658/1.677, 1-1.323/1.403, 1-1.596/1.654, 1-1.303/1.419, 1-1.276/1.344,1-2.286/2.446, 1-1.062/1.079],
            [ 1-1.618/1.640,  1-1.313/1.362, 1-1.528/1.548, 1-1.231/1.333, 1-1.389/1.499, 1-2.496/2.636, 1-1.102/1.176],
            [ 1-1.648/1.696,  1-1.365/1.391, 1-1.667/1.737, 1-1.247/1.286, 1-1.410/1.527,1-2.936/3.069, 1-1.099/1.305]])
    ]
    return data

def to_percent(temp, position):
    return '%1.0f'%(10*temp) + '%'

if __name__ == '__main__':

    N = 7
    theta = radar_factory(N, frame='polygon')

    data = example_data()
    spoke_labels = data.pop(0)

    fig, axs = plt.subplots(figsize=(9, 9), nrows=1, ncols=1,
                            subplot_kw=dict(projection='radar'))
    fig.subplots_adjust(wspace=0.25, hspace=0.20, top=0.85, bottom=0.05)

    colors = ['b', 'r','g']
    # Plot the four cases from the example data on separate axes
    # for ax, (title, case_data) in zip(axs.flat, data):
    #     ax.set_rgrids([0.1, 0.4, 0.6, 0.8, 1.0,])
    #     ax.set_title(title, weight='bold', size='medium', position=(0.5, 1.1),
    #                  horizontalalignment='center', verticalalignment='center')
    #     for d, color in zip(case_data, colors):
    #         ax.plot(theta, d, color=color)
    #         ax.fill(theta, d, facecolor=color, alpha=0.25, label='_nolegend_')
        # ax.set_varlabels(spoke_labels)
    case_data = data[0][1]
    axs.set_rgrids(radii = [0.05,  0.1,0.15,0.2],labels=[f'$5\%$',f'$10\%$',f'$15\%$',f'$20\%$'])

    # axs.set_title('Refinement results on seven datasets', weight='bold', size='medium', position=(0.5, 1.1),
    #                 horizontalalignment='center', verticalalignment='center')
    for d, color in zip(case_data, colors):
        axs.plot(theta, d, color=color)
        axs.fill(theta, d, facecolor=color, alpha=0.1, label='_nolegend_')
    # add legend relative to top-left plot
    labels = ('Arima', 'ETS','Theta')
    legend = axs.legend(labels, loc=(0.8, .9),
                        labelspacing=0.1, fontsize=15)
    axs.set_varlabels(spoke_labels)
    # fig.text(0.5, 0.965, 'Refinement results on seven datasets',
    #          horizontalalignment='center', color='black', weight='bold',
    #          size='large')
    plt.tight_layout()
    plt.show()