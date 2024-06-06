import {Component} from "@angular/core"
@Component({
    selector : "page-not-found",
    templateUrl : "pagenotfound.component.html",
    styleUrls: ["./pagenotfound.component.css"]
})
export class PageNotFoundComponent{
    message : String = "Page not found"
}