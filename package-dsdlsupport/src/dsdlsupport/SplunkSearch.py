import os
import time

import ipywidgets as widgets
import pandas as pd
import splunklib.client as splunk_client
import splunklib.results as splunk_results
from IPython.display import display


class SplunkSearch(object):
    def __init__(self,
                 host=None,
                 port=None,
                 token=None,
                 search='| makeresults count=10 \n| streamstats c as i \n| eval s = i%3 \n| eval feature_{s}=0 \n| foreach feature_* [eval <<FIELD>>=random()/pow(2,31)]'):

        # internal refernce to splunk service object
        self._service = None
        self._job = None
        self._results = []

        self.host = host
        self.port = port
        self.token = token

        if "splunk_access_enabled" in os.environ:
            access_enabled = os.environ["splunk_access_enabled"]
            if access_enabled=="1":
                self.host = os.environ["splunk_access_host"]
                self.port = os.environ["splunk_access_port"]
                self.token = os.environ["splunk_access_token"]

        # generate widgets UI components
        ui = {}
        ui['spl'] = widgets.Textarea(
            value=search,
            placeholder='index=_internal | stats count by sourcetype',
            description='search',
            layout=widgets.Layout(width='90%', height='90px'),
            disabled=False
        )
        ui['earliest'] = widgets.Text(
            value='-15m@m',
            placeholder='-15m@m',
            description='earliest',
            layout=widgets.Layout(width='90%'),
            disabled=False
        )
        ui['latest'] = widgets.Text(
            value='now',
            placeholder='now',
            description='latest',
            layout=widgets.Layout(width='90%'),
            disabled=False
        )
        ui['button'] = widgets.Button(
            description='search',
            disabled=False,
            button_style='', # 'success', 'info', 'warning', 'danger' or ''
            tooltip='Start your splunk search',
            layout=widgets.Layout(width='90%'),
            icon='search' # (FontAwesome names without the `fa-` prefix)
        )
        ui['button'].on_click(self.search_button_clicked)
        ui['progress1'] = widgets.FloatProgress(value=0.0, min=0.0, max=100.0, description="search", layout=widgets.Layout(width='100%'), style={'bar_color': '#ed028b'})
        ui['progress2'] = widgets.FloatProgress(value=0.0, min=0.0, max=1.0, description="results", layout=widgets.Layout(width='100%'), style={'bar_color': '#f56a00'})
        ui['resultinfo'] = widgets.Text(
            value='',
            placeholder='',
            description='result info',
            layout=widgets.Layout(width='100%'),
            disabled=True
        )

        # generate the widgets in a box layout
        self.widgets = widgets.VBox([
            widgets.HBox([ui['spl'], widgets.VBox([ui['button'], ui['earliest'], ui['latest']], layout=widgets.Layout(width='20%'))]),
            ui['progress1'],
            ui['progress2'],
            ui['resultinfo']
        ])

        # keep the reference to the UI widgets
        self.ui = ui

        # display the UI
        display(self.widgets)


    def search_button_clicked(self, button_event):
        self.ui['button'].disabled = True
        self.ui['progress1'].description = "search"
        self.ui['progress2'].description = "results"
        self.ui['progress1'].value = 0.0
        self.ui['progress2'].value = 0.0
        resultset = self.search(
            query=self.ui['spl'].value,
            earliest=self.ui['earliest'].value,
            latest=self.ui['latest'].value,
        )
        self.ui['button'].disabled = False
        self._results = resultset

    @property
    def service(self):
        if self._service is not None:
            return self._service
        self._service = splunk_client.connect(host=self.host, port=self.port, token=self.token)
        return self._service

    def search(self, query, earliest="-15m@m", latest="now"):
        # preprocess the SPL query
        query_cleaned = query.strip()
        if len(query_cleaned)==0:
            self.ui['resultinfo'].value = "search error: %s" % "empty search string. please enter valid SPL."
            return
        elif query_cleaned[0]=='|':
            # assume a generating search command and do nothing
            pass
        elif query_cleaned.startswith("search ") or query_cleaned.startswith("search\n"):
            # assume the search keyword is already there
            pass
        else:
            # add search keyword before the SPL
            query_cleaned="search "+query_cleaned

        resultset = []
        resultinfo = "An error occurred"
        try:
            # create a search job in splunk
            job = self.service.jobs.create(
                    query_cleaned,
                    earliest_time=earliest,
                    latest_time=latest,
                    adhoc_search_level="smart",
                    search_mode="normal")
            self._job = job
            self.ui['progress1'].description = "searching..."
            self.ui['progress1'].max = 100.0
            doneProgress = 0.0
            while not job.is_done():
                time.sleep(0.1)
                doneProgress = float(job["doneProgress"])*100.0
                self.ui['progress1'].value = doneProgress
            self.ui['progress1'].value = self.ui['progress1'].max
            self.ui['progress1'].description = "search done"

            resultCount = int(job.resultCount)
            processed = 0
            offset = 0
            self.ui['progress2'].description = "loading..."
            self.ui['progress2'].max = float(resultCount)
            diagnostic_messages = []
            while processed < resultCount:
                for event in splunk_results.JSONResultsReader(job.results(output_mode='json', offset=offset, count=0)):
                    if isinstance(event, splunk_results.Message):
                        # Diagnostic messages may be returned in the results
                        diagnostic_messages.append(event.message)
                        #print('%s: %s' % (event.type, event.message))
                    elif isinstance(event, dict):
                        # Normal events are returned as dicts
                        resultset.append(event)
                        #print(result)
                    processed += 1
                    self.ui['progress2'].value = float(processed)
                offset = processed
            self.ui['progress2'].value = self.ui['progress2'].max
            self.ui['progress2'].description = "loading done"

            resultinfo = "search completed with %d results." % int(job.resultCount)
            # raw dump further infos
            if len(diagnostic_messages)>0:
                resultinfo += str(diagnostic_messages)
            if len(job.state.content.messages)>0:
                resultinfo += str(job.state.content.messages)
        except Exception as e:
            resultinfo = "An error occurred: " + str(e) + " - " + resultinfo
        finally:
            self.ui['resultinfo'].value = resultinfo
        return resultset

    def search_logs(self):
        if self._job is not None:
            for line in self._job.searchlog():
                print(str(line))

    def as_df(self):
        return pd.DataFrame(self._results)