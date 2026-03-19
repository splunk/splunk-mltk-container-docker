const appName = window.location.pathname.match(/..-..\/app\/(?<app>[^\/]+)/).groups.app;
require([
    'jquery',
    'splunkjs/mvc',
    'splunkjs/mvc/simplexml/ready!'
 ], function ($, mvc,) {
        var submittedTokens = mvc.Components.get('submitted');
        submittedTokens.on('change:plot_matrix', function(model, tokBASE64, options) {
            var tokHTMLJS=submittedTokens.get('plot_matrix');
            if(tokHTMLJS!==undefined) {
                document.getElementById('plot_matrix').src = 'data:image/jpeg;base64,' + tokHTMLJS;
            }
        });
        submittedTokens.on('change:plot_pairplot', function(model, tokBASE64, options) {
            var tokHTMLJS=submittedTokens.get('plot_pairplot');
            if(tokHTMLJS!==undefined) {
                document.getElementById('plot_pairplot').src = 'data:image/png;base64,' + tokHTMLJS;
            }
        });
});