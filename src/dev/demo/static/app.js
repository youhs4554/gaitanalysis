var FunctionOne = function () {
    // create a deferred object
    var r = $.Deferred();
  
    // do whatever you want (e.g. ajax/animations other asyc tasks)
    $(document).ajaxStart(function(){
      $('#spinner').show();
   }).ajaxStop(function(){
      $('#spinner').hide();
   });


   $.ajax({
    type : 'GET',
    url : "/run",
    data : {start: "True"},
    success : function(data){
      if (data == '1'){
          r.resolve();
          // redirect to status page
          var url = window.location.href + 'stat';
          window.location.assign(url);
      }
    },
     error : function(xhr, status, error) {
      // redirect to status page
      //var url = window.location.href
      //window.location.assign(url)
      console.log('fuck!')
     }
    });


  };

  // call FunctionOne and use the `done` method
  // with `FunctionTwo` as it's parameter
  FunctionOne()