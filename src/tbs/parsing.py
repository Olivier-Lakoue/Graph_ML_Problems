from lxml import etree
from glob import glob
import re
import pandas as pd
import os

class Regulations():
    def __init__(self, folder):
        self.reg_files = glob(folder+'*.xml')
        self.filtered_reg_files = self.exclude_SI()

    def exclude_SI(self):
        return [r for r in self.reg_files if re.match(r'^((?!SI).)*$', r)]

    def parse(self):
        titles = []
        files = []
        repealed = []
        references = []

        for i, f in enumerate(self.filtered_reg_files):
            # parse xml file
            doc = etree.parse(f)
            # get the file name
            fname = os.path.basename(f)
            try:
                title = doc.xpath('//LongTitle')[0].text.lower()
            except:
                title = ''
            titles.append(title)
            files.append(fname)
            try:
                repeal = doc.xpath('//Repealed')[0].text
            except:
                repeal = ''
            repealed.append(repeal)

            try:
                xref = doc.xpath('//XRefExternal')
                xref = [x.text.lower() for x in doc.xpath('//XRefExternal') if x.text.lower() != title]
            except:
                xref = []
            references.append(xref)

        self.df = pd.DataFrame({'titles': titles, 'files': files, 'repealed': repealed, 'references': references})
        self.df['class_type'] = [self.title_classification(x) for x in self.df.titles.values]
        return self.df

    def title_classification(self, title):
        """
        Return the identified class of regulation.
        :param title: string
        :return: string
        """
        pattern = re.compile(
            r'(regulation|order|rule|procla|direction|fees|permit|notice|by-law|authorization|list|guideline)')
        if pattern.search(title):
            res = pattern.search(title).group(0)
        else:
            res = 'other'
        return res

class Acts():
    def __init__(self, folder):
        self.reg_files = glob(folder + '*.xml')

    def parse(self):
        titles = []
        short_titles = []
        files = []
        repealed = []
        references = []
        for i, f in enumerate(self.reg_files):
            doc = etree.parse(f)
            filename = os.path.basename(f)
            files.append(filename)
            #     print(filename)
            try:
                short = doc.xpath('//ShortTitle')[0].text.lower()
            except:
                short = ''
            short_titles.append(short)

            try:
                repeal = doc.xpath('//Repealed')[0].text
            except:
                repeal = ''
            repealed.append(repeal)

            try:
                title = doc.xpath('//LongTitle')[0].text.lower()
            except:
                title = ''
            titles.append(title)

            try:
                # get refs and remove self ref
                Xref = [x.text.lower() for x in doc.xpath('//XRefExternal') if x.text.lower() != short]
            except:
                Xref = []
            references.append(Xref)

        self.df = pd.DataFrame({'titles': titles, 'short_title': short_titles, 'files': files, 'repealed': repealed,
                           'references': references})
        return self.df
