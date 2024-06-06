import { NgModule } from '@angular/core';
import { BrowserModule } from '@angular/platform-browser';
import {FormsModule} from "@angular/forms"
import { AppRoutingModule } from './app-routing.module';
import { AppComponent } from './app.component';
import {NavbarComponent} from "./Navbar/navbar.component"
import {LoginComponent} from "./Login/login.component"
import {HttpClientModule} from "@angular/common/http"
import {LoginSerice} from "./service/login.service"
import {PageNotFoundComponent} from "./pagenotfound/pagenotfound.component"
import { VideoUploadComponent } from './videoupload/videoupload.component';
import { VideoProcessService } from './service/videoprocess.service';
import {PridComponent} from "./prid/prid.component"
import { PridDemoComponent } from './prid_demo/prid_demo.component';
import { ImageCropperModule } from 'ngx-image-cropper';
@NgModule({
  declarations: [
    AppComponent,
    NavbarComponent,
    LoginComponent,
    PageNotFoundComponent,
    VideoUploadComponent,
    PridComponent,
    PridDemoComponent
  ],
  imports: [
    BrowserModule,
    AppRoutingModule,
    FormsModule,
    HttpClientModule,
    ImageCropperModule
  ],
  providers: [LoginSerice,VideoProcessService],
  bootstrap: [AppComponent]
})
export class AppModule { }
