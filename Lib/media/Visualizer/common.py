def format_with_suffix(num, signed=True, percent=False):
        suffixes = {
            1: '',
            1e3: 'k',
            1e6: 'M',
            1e9: 'G',
            1e12: 'T'
        }
        suff_list = list(suffixes.keys())
        sign = '+' if num > 0 else ''
        if not signed:
            sign = ""
        num_format = '{}{:.1f}{}'
        simple_format = '{}{:.3f}'
        if not num:
            return simple_format.format(sign, num)
        
        if percent:
            return f'{num:.1%}'

        if num in suffixes:
            return num_format.format(sign, 1.0, suffixes[num])

        if suff_list[0] > abs(num):
            return simple_format.format(sign, num)

        for ind, prefix in enumerate(suffixes):
            if prefix > abs(num):
                suffix = suffixes[suff_list[ind - 1]]
                return num_format.format(sign, float(num) / suff_list[ind - 1], suffix)

            
def make_link(link, metric, metrics_dict):
        first, second = link.split('metric=')
        third = "&".join(second.split('&')[1:])
        link = first + "metric=" + str(metrics_dict[metric]) + "&" + third
        text = "Cсылка на M42 с описанием разреза"
        return f"""<a href="{link}">{text}</a>"""