$(function() {
  var vid = document.getElementById("vid");
});

getStartTime = () => {
  var currentTime = vid.currentTime;
  $("#startTime").html(currentTime);
};

getEndTime = () => {
  var currentTime = vid.currentTime;
  var startTime = $("#startTime").text();
  if (startTime == "") {
    alert("Set startTime first!");
  } else if (startTime >= currentTime) {
    alert("endTime should be bigger than startTime!");
  }
  $("#endTime").html(currentTime);
};

run_api = () => {
  $.ajax({
    type: "GET",
    url: "/run",
    data: {
      start: "True",
      startTime: $("#startTime").text(),
      endTime: $("#endTime").text()
    },
    success: function(data) {
      if (data == "1") {
        // redirect to status page
        var url = window.location.origin + "/stat";
        window.location.assign(url);
      }
    },
    error: function(xhr, status, error) {
      // redirect to status page
      //var url = window.location.href
      //window.location.assign(url)
      console.log("fuck!");
    }
  });
};
