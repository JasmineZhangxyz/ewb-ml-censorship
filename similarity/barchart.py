import matplotlib.pyplot as plt


def bar_chart(social_percentage, political_percentage):
    """
    This function shows the figure of a bar chart with the percentage of Social and Political censored words for a
    given company on the x-axis. The y-axis does not exceed 1.

    :param social_percentage: the percentage of words censored by a company that is labelled political.
    :param political_percentage: the percentage of words censored by a company that is labelled political.
    :return: figure of bar chart.
    """

    fig = plt.figure()
    ax = fig.add_axes([0,0,1,1])
    ax.set_xlabel("Label of Word")
    ax.set_ylabel("Percentage in Company")
    label = ['Social', 'Political']
    percentage = [social_percentage * 100, political_percentage * 100]
    ax.bar(label, percentage)
    plt.show()
